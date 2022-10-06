import torch
import torch.nn as nn
import numpy as np
import wandb
import json



import logging
logger = logging.getLogger(__name__)


class Reshape(nn.Module):
    '''
    past_key_values (`Tuple[Tuple[torch.Tensor]]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of length `config.n_layers`, containing tuples of tensors of shape `(batch_size, num_heads,
            sequence_length, embed_size_per_head)`).
    '''
    def __init__(self, arg_dict):
        super(Reshape, self).__init__()
        self.seq_len = arg_dict['seq_len']
        self.num_layer = arg_dict['num_layer']
        self.hidden_size = arg_dict['hidden_size']
        self.num_head = arg_dict['num_head']
    def forward(self, x):
        batch_size = x.shape[0]
        assert self.hidden_size % self.num_head == 0
        embed_size_per_head = self.hidden_size//self.num_head
        x = x.view(batch_size, self.num_layer, 2, self.num_head, self.seq_len, embed_size_per_head).permute(1,2,0,3,4,5)
        past_key_values = []
        for i in range(self.num_head):
            past_key_values.append((x[i][0],x[i][1],))
        assert past_key_values[0][0].requires_grad == True
        return tuple(past_key_values)



class AE(nn.Module):
    """
    AE with Decoder Fixed.
    We adopt connection method from prompt tuning.
    """
    #_keys_to_ignore_on_load_missing = [r"latent_classify_head\.\d+\.weight", r"latent_classify_head\.\d+\.bias"]
    def __init__(self, encoder, decoder, args): # 
        super(AE, self).__init__()
        self.encoder = encoder #BertModel
        self.decoder = decoder #GPT2LMHeadModel

        self.encoder_config = encoder.config
        self.encoder_hidden_size = self.encoder_config.hidden_size

        self.decoder_config = decoder.config
        self.decoder_num_layer = self.decoder_config.n_layer
        self.decoder_hidden_size = self.decoder_config.n_embd
        self.decoder_num_head = self.decoder_config.n_head

        self.losslist = None
        self.latent_classify_head = None

        

        self.args = args
        self.latent_size = args.latent_size
        self.seq_len_per_latent = args.seq_len_per_latent
        self.latent_num = args.latent_num
        self.seq_len = self.latent_num * self.seq_len_per_latent
        if 'variation' in args:
            self.variation = args.variation
        else:
            self.variation = 0


        ## connector: 
        # 1. from Bert hidden units to the latent space
        # 2. convert latent space to `past_key_values' in GPT
        # [batch_size, bert_hidden_size] -> [batch_size, latent_num * latent_size]
        # -> [batch_size, latent_num, decoder_layer * len([key,value]) * gpt_hidden_size]
        # -> (num_layer* (len([key,value])* tensor[batch_size, num_head, seq_len, embed_size_per_head]))
        '''
        self.trans = torch.nn.Sequential(
                    torch.nn.Linear(self.encoder_hidden_size, self.latent_num * self.latent_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(self.latent_num * self.latent_size, self.seq_len * self.decoder_num_layer * 2 * self.decoder_hidden_size),
                    Reshape({'seq_len':self.seq_len, 'num_layer':self.decoder_num_layer, 'hidden_size':self.decoder_hidden_size, 'num_head':self.decoder_num_head})
                    )
        '''
        self.trans1 = torch.nn.Sequential(
            torch.nn.Linear(self.encoder_hidden_size, self.latent_num * self.latent_size),
            torch.nn.Tanh(),
            nn.Dropout(self.decoder_config.attn_pdrop)#added
        )
        self.trans2 = torch.nn.Sequential(
            torch.nn.Linear(self.latent_num * self.latent_size, self.seq_len * self.decoder_num_layer * 2 * self.decoder_hidden_size),
            nn.Dropout(self.decoder_config.attn_pdrop),#added
            Reshape({'seq_len':self.seq_len, 'num_layer':self.decoder_num_layer, 'hidden_size':self.decoder_hidden_size, 'num_head':self.decoder_num_head})
        )
    

    def fix_decoder(self):
        '''
        Fix the decoder to work as prefix tuning.
        '''
        self.decoder.eval()
        for param in self.decoder.parameters():
            param.requires_grad = False


    def connect(self, encoder_output, variation=0):
        '''
        Connect encoder to decoder and get the latent representation.
        '''
        tmp_latent = self.trans1(encoder_output)
        eps = torch.zeros_like(tmp_latent).normal_(std=variation).to(tmp_latent.device)
        past_key_values = self.trans2(tmp_latent + eps)
        latent = tmp_latent.view(-1, self.latent_num, self.latent_size)
        return past_key_values, latent


    def sparse_loss(self, latent, dim=None):
        '''
        Increase the sparsity.
        '''
        if len(latent) == 3 and dim is None:
            raise Exception('Expect latent to be dim 2.')
        loss_func = nn.L1Loss(reduction='mean')
        batch_size = latent.shape[0]
        if dim is not None:
            tmp_latent = latent[:,dim,:].squeeze()
            average = torch.sum(tmp_latent, dim=0)/batch_size
            loss = loss_func(latent, average.expand(batch_size, -1))
        else:
            average = torch.sum(latent, dim=0)/batch_size
            loss = loss_func(latent, average.expand(batch_size, -1))
        return -loss

    def contrasitive_loss(self, latent1, latent2, loss_func=nn.SmoothL1Loss(reduction='mean'), dim=None):
        '''
        Increase the distance between latent1 and latent2.
        loss_func: nn.L1Loss, nn.SmoothL1Loss, nn.MSELoss, ...
        '''
        if dim is not None:
            loss = loss_func(latent1[:,dim,:].squeeze(), latent2[:,dim,:].squeeze())
        else:
            loss = loss_func(latent1, latent2)
        return -1 * loss
    
    def latent_classify_loss(self, latent, pos_label, neg_labels, head_index=None):
        if len(latent.shape) == 3:
            latent = latent.view(-1, self.latent_num*self.latent_size)
        if self.latent_classify_head_type == 'single':
            probs = torch.softmax(self.latent_classify_head(latent), dim=-1)
            batch_size, class_num = probs.shape
            loss = 0
            neg_len = neg_labels.shape[-1]
            
            for i in range(batch_size):
                pos_prob = probs[i, pos_label[i]]
                if pos_prob < 1/self.head_num:
                    loss += torch.log(pos_prob)
                loss += torch.log(1 - probs[i, neg_labels[i]]).sum()

            return -1 * loss / (batch_size * (neg_len+1))
        elif self.latent_classify_head_type == 'multiple':
            if head_index is None:
                print("UserWarning: head_index not set for multiple classifier head, default to 0")
                head_index = 0
            device = latent.device
            logits = self.latent_classify_head[head_index](latent)
            loss = torch.nn.functional.cross_entropy(logits, pos_label.to(device))
            return loss
        else:
            raise Exception('Wrong latent classifier head type.')

    def aspect_gap_loss(self, latent, head_index):
        if len(latent.shape) == 3:
            latent = latent.view(-1, self.latent_num * self.latent_size)

        mean_latent = torch.mean(latent, dim=0)
        loss = None
        for i in range(self.aspect_head_num):
            if i != head_index and self.aspect_gap_head[i] is not None:
                if loss is None:
                    loss = torch.nn.functional.mse_loss(mean_latent, self.aspect_gap_head[i]) * self.aspect_gap_loss_amplification
                else:
                    loss += torch.nn.functional.mse_loss(mean_latent, self.aspect_gap_head[i]) * self.aspect_gap_loss_amplification
        self.set_aspect_gap_head(mean_latent, head_index)
        return loss

    def set_losslist(self, 
        losslist:dict,
        latent_classify_args={'head_num':1, 'class_num_per_head':2,'mid_size':128,'head_type':'single'},
        aspect_gap_args={'head_num':2, 'amplification':5}
        ):
        '''
        losslist:
            Sample: {'contrasitive_loss': 0.001, 'sparse_loss': 0.001, 'latent_classify_loss':0.1, 'aspect_gap_loss':0.1}
        '''
        self.losslist = losslist
        if 'latent_classify_loss' in losslist:
            self.head_num = 1
            class_num_per_head = 2
            mid_size = 128
            head_type = 'single'
            if latent_classify_args is not None:
                if 'head_num' in latent_classify_args:
                    self.head_num = latent_classify_args['head_num']
                if 'class_num_per_head' in latent_classify_args:
                    class_num_per_head = latent_classify_args['class_num_per_head']
                if 'mid_size' in latent_classify_args:
                    mid_size = latent_classify_args['mid_size']
                if 'head_type' in latent_classify_args:
                    head_type = latent_classify_args['head_type']
            
            self.set_latent_classify_head(head_num=self.head_num, class_num_per_head=class_num_per_head, mid_size=mid_size, head_type=head_type)
    
            self.latent_classify_head_type=head_type
        
        if 'aspect_gap_loss' in losslist:
            if 'latent_classify_loss' in losslist:
                if self.latent_classify_head_type == 'multiple':
                    self.aspect_head_num = self.head_num
                elif self.latent_classify_head == 'single':
                    print('set aspect head num to {aspect_head_num}.')
                    self.aspect_head_num = aspect_gap_args['head_num']
            else:
                print('set aspect head num to {aspect_head_num}.')
                self.aspect_head_num = aspect_gap_args['head_num']
            
            self.aspect_gap_loss_amplification = aspect_gap_args['amplification']

            self.aspect_gap_head = [None for i in range(self.aspect_head_num)]

    def set_latent_classify_head(self, head_num=1, class_num_per_head=2, mid_size=128, head_type='single'):
        if head_type == 'single':
            self.latent_classify_head = nn.Sequential(
                nn.Linear(self.latent_num * self.latent_size, mid_size),
                nn.ReLU(),
                nn.Linear(mid_size, class_num_per_head * head_num)
            )
        elif head_type == 'multiple':
            if type(class_num_per_head) is list:
                self.latent_classify_head = nn.ModuleList([
                        nn.Sequential(
                        nn.Linear(self.latent_num * self.latent_size, mid_size),
                        nn.ReLU(),
                        nn.Linear(mid_size, head_num)
                    ) for head_num in class_num_per_head]
                )
            else:
                self.latent_classify_head = nn.ModuleList([
                        nn.Sequential(
                        nn.Linear(self.latent_num * self.latent_size, mid_size),
                        nn.ReLU(),
                        nn.Linear(mid_size, class_num_per_head)
                    ) for i in range(head_num)]
                )
    
    def set_aspect_gap_head(self, latent, head_index):
        if len(latent.shape) == 3:
            latent = latent.view(-1, self.latent_num * self.latent_size)
        
        if len(latent.shape) == 2:
            mean_latent = torch.mean(latent.detach(), dim=0)
            if self.aspect_gap_head[head_index] is not None:
                assert self.aspect_gap_head[head_index].shape == mean_latent.shape
            self.aspect_gap_head[head_index] = mean_latent
        elif len(latent.shape) == 1:
            if self.aspect_gap_head[head_index] is not None:
                assert self.aspect_gap_head[head_index].shape == latent.shape
            self.aspect_gap_head[head_index] = latent.detach()

        

    
    def forward(self,
        encoder_input_ids,
        encoder_attention_mask,
        encoder_token_type_ids,
        decoder_input_ids,
        decoder_attention_mask,
        adv_input_ids=None,
        adv_attention_mask=None,
        adv_token_type_ids=None,
        pos_label=None,
        neg_labels=None,
        variation=None,
        head_index=None
        ):
        '''
        Forward method for training which returns a reconstruction loss.
        Args:
            encoder_input_ids,
            encoder_atention_mask,
            encoder_token_type_ids:
                Outputs of BertTokenizer(List of Strings, return_tensors='pt', padding=True)
            decoder_input_ids,
            decoder_attention_mask:
                Outputs of GPT2Tokenizer(List of Strings, return_tensors='pt', padding=True)
            adv_input_ids,
            adv_attention_mask,
            adv_token_type_ids:
                Adversarial text

        '''
        if len(encoder_input_ids.shape) == 3:
            encoder_input_ids = encoder_input_ids.view(encoder_input_ids.shape[1], encoder_input_ids.shape[2])
            encoder_attention_mask = encoder_attention_mask.view(encoder_attention_mask.shape[1], encoder_attention_mask.shape[2])
            encoder_token_type_ids = encoder_token_type_ids.view(encoder_token_type_ids.shape[1], encoder_token_type_ids.shape[2])
            decoder_input_ids = decoder_input_ids.view(decoder_input_ids.shape[1], decoder_input_ids.shape[2])
            decoder_attention_mask = decoder_attention_mask.view(decoder_attention_mask.shape[1], decoder_attention_mask.shape[2])
            if head_index is not None:
                head_index = head_index.item()
            if adv_input_ids is not None:
                adv_input_ids = adv_input_ids.view(adv_input_ids.shape[1], adv_input_ids.shape[2])
            if adv_attention_mask is not None:
                adv_attention_mask = adv_attention_mask.view(adv_attention_mask.shape[1], adv_attention_mask.shape[2])
            if adv_token_type_ids is not None:
                adv_token_type_ids = adv_token_type_ids.view(adv_token_type_ids.shape[1], adv_token_type_ids.shape[2])
            if pos_label is not None:
                pos_label = pos_label.view(pos_label.shape[1])
            if neg_labels is not None:
                neg_labels = neg_labels.view(neg_labels.shape[1], neg_labels.shape[2])

        if variation is None:
            variation = self.variation
        batch_size = decoder_input_ids.shape[0]
        infix_attn = torch.ones(batch_size, self.seq_len).bool().to(decoder_input_ids.device)
        decoder_attention_mask = torch.cat([infix_attn, decoder_attention_mask], dim=1)

        encoder_output = self.encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask, token_type_ids=encoder_token_type_ids, return_dict=True).pooler_output
        past_key_values, latent = self.connect(encoder_output, variation)
        outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, labels=decoder_input_ids, past_key_values=past_key_values, return_dict=True)
        lm_loss = outputs.loss
        # lm_logits = outputs.logits
        
        #labels = decoder_input_ids
        # Shift so that tokens < n predict n
        #shift_logits = lm_logits[..., self.seq_len:-1, :].contiguous()#########lm_logits[..., :-1, :] -> lm_logits[..., self.seq_len:-1, :]
        #shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        #loss_fct = torch.nn.CrossEntropyLoss()
        #lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        #logits = outputs.lm_logits
        loss = 0
        loss_detail = {"lm_loss":lm_loss.detach()}
        w = 1
        if self.losslist is not None:
            if 'contrasitive_loss' in self.losslist:
                if adv_input_ids is None:
                    raise Exception('Expect adversarial inputs for contrasitive loss.')
                adv_encoder_output = self.encoder(input_ids=adv_input_ids, attention_mask=adv_attention_mask, token_type_ids=adv_token_type_ids, return_dict=True).pooler_output
                adv_latent = self.trans1(adv_encoder_output)
                adv_loss = self.contrasitive_loss(latent, adv_latent)
                #TO DO: change arg `dim' in the future
                loss += adv_loss * self.losslist['contrasitive_loss']
                w -= self.losslist['contrasitive_loss']
                loss_detail["contrasitive_loss"] = adv_loss.detach()
            
            if 'sparse_loss' in self.losslist:
                spa_loss = self.sparse_loss(latent)
                #TO DO: change arg `dim' in the future
                loss += spa_loss * self.losslist['sparse_loss']
                w -= self.losslist['sparse_loss']
                loss_detail["sparse_loss"] = spa_loss.detach()
            
            if 'latent_classify_loss' in self.losslist:
                lac_loss = self.latent_classify_loss(latent, pos_label, neg_labels, head_index)
                if lac_loss.detach().item() < 0.1:
                    loss += lac_loss * 0.05
                    w -= 0.05
                else:
                    loss += lac_loss * self.losslist['latent_classify_loss']
                    w -= self.losslist['latent_classify_loss']
                loss_detail["latent_classify_loss"] = lac_loss.detach()


            if 'aspect_gap_loss' in self.losslist:
                agp_loss = self.aspect_gap_loss(latent, head_index)
                if agp_loss is not None:
                    loss += agp_loss * self.losslist['aspect_gap_loss']
                    w -= self.losslist['aspect_gap_loss']
                    loss_detail["aspect_gap_loss"] = agp_loss.detach()
            
            wandb.log(loss_detail)
        if w < 0:
            w = 1
        
        loss += w * lm_loss
        
        return loss, latent, loss_detail
    

    def encode(self,
        encoder_input_ids,
        encoder_attention_mask=None,
        encoder_token_type_ids=None,
        ):
        '''
        Encode the input text and get the latent representation
        '''
        device = next(self.parameters()).device
        encoder_input_ids = encoder_input_ids.to(device)
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(device)
        if encoder_token_type_ids is not None:
            encoder_token_type_ids = encoder_token_type_ids.to(device)
        encoder_output = self.encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask, token_type_ids=encoder_token_type_ids, return_dict=True).pooler_output
        past_key_values, latent = self.connect(encoder_output)
        return latent, encoder_output, past_key_values


    def generate(
        self,
        input_latent,
        input_ids=None,
        attention_mask=None,
        batch_size=None,
        variation=None,
        min_len=30,
        max_len=50,
        do_sample=True,
        topk=5,
        topp=0.9,
        lp=1,
        rp=1.0,
        use_cache=True):
        '''
        Generate text with given latent represention.
        '''
        device = next(self.parameters()).device
        input_latent = input_latent.to(device)
        

        if len(input_latent.shape) == 3:
            tmp_batch_size, latent_num, latent_size = input_latent.shape
            input_latent = input_latent.view(tmp_batch_size, latent_num * latent_size)
        elif len(input_latent.shape) != 2:
            raise Exception('Shape of input_latent is expected to be [batch_size, latent_num, latent_size] \
                or [batch_size, latent_num * latent_size]')


        if batch_size is None:
            batch_size = input_latent.shape[0]
            if input_ids is not None:
                if input_ids.shape[0] > batch_size:
                    if batch_size == 1:
                        batch_size = input_ids.shape[0]
                        input_latent = input_latent.expand(batch_size, -1)
                    else:
                        raise Exception('Batch size of input_latent and input_ids mismatched')
                elif input_ids.shape[0] < batch_size and input_ids.shape[0] == 1:
                    input_ids = input_ids.expand(batch_size, -1)
                


        if input_latent.shape[0] < batch_size:
            input_latent.expand(batch_size, -1)


        if variation is not None:
            eps = torch.zeros_like(input_latent).normal_(std=variation).to(input_latent.device)
            input_latent = input_latent + eps        


        past_key_values = self.trans2(input_latent)

        if input_ids is None:
            input_ids = self.decoder.generate(input_ids=torch.LongTensor([[50256]]*batch_size).to(device), max_length=3, do_sample=True)[:,1:]
            attention_mask = torch.ones(batch_size, 2).bool()
        else:
            input_ids = input_ids.to(device)
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, 2).bool()

        cur_len = input_ids.shape[1]
        infix_attn = torch.ones(batch_size, self.seq_len).bool().to(device)
        attention_mask = torch.cat([infix_attn, attention_mask.to(device)], dim=-1)

        if cur_len < 1:
            raise Exception('input length error')
        if cur_len == 1:
            result = self.decoder.generate(input_ids=input_ids, past=past_key_values, attention_mask=attention_mask, repetition_penalty=rp,\
                do_sample=do_sample, top_k=topk, top_p=topp, length_penalty=lp, max_length=max_len, min_length=min_len, use_cache=use_cache)
        else:
            past_key_values = self.decoder(input_ids=input_ids[:,:-1], attention_mask=attention_mask[:,:-1], past_key_values=past_key_values, return_dict=True, use_cache=True).past_key_values
            result = self.decoder.generate(input_ids=input_ids, past=past_key_values, attention_mask=attention_mask, repetition_penalty=rp,\
                do_sample=do_sample, top_k=topk, top_p=topp, length_penalty=lp, max_length=max_len, min_length=min_len, use_cache=use_cache)

        return result


    def reconstruct(self,
        encoder_input_ids,
        decoder_input_ids=None,
        encoder_attention_mask=None,
        encoder_token_type_ids=None,
        decoder_attention_mask=None,
        do_sample=True,
        max_len=50,
        min_len=30,
        topk=5,
        topp=0.9,
        lp=1.0,
        use_cache=True):
        '''
        Reconstruct input text.
        '''
        device = next(self.parameters()).device
        batch_size = encoder_input_ids.shape[0]
        encoder_input_ids = encoder_input_ids.to(device)
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(device)
        if encoder_token_type_ids is not None:
            encoder_token_type_ids = encoder_token_type_ids.to(device)
        encoder_output = self.encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask, token_type_ids=encoder_token_type_ids, return_dict=True).pooler_output
        
        past_key_values, latent = self.connect(encoder_output)
        if decoder_input_ids is None:
            decoder_input_ids = self.decoder.generate(input_ids=torch.LongTensor([[50256]]*batch_size).to(device), max_length=3, do_sample=True)[:,1:]
            decoder_attention_mask = torch.ones(batch_size, 2).bool()
        else:
            decoder_input_ids = decoder_input_ids.to(device)
            if decoder_attention_mask is None:
                decoder_attention_mask = torch.ones(batch_size, 2).bool()
        
        cur_len = decoder_input_ids.shape[1]
        
        infix_attn = torch.ones(batch_size, self.seq_len).bool().to(device)
        decoder_attention_mask = torch.cat([infix_attn, decoder_attention_mask.to(device)], dim=-1)

        if cur_len < 1:
            raise Exception('input length error')
        if cur_len == 1:
            result = self.decoder.generate(input_ids=decoder_input_ids, past=past_key_values, attention_mask=decoder_attention_mask,\
                do_sample=do_sample, top_k=topk, top_p=topp, length_penalty=lp, max_length=max_len, min_length=min_len, use_cache=use_cache)
        else:
            past_key_values = self.decoder(input_ids=decoder_input_ids[:,:-1], attention_mask=decoder_attention_mask[:,:-1], past_key_values=past_key_values, return_dict=True, use_cache=True).past_key_values
            result = self.decoder.generate(input_ids=decoder_input_ids, past=past_key_values, attention_mask=decoder_attention_mask,\
                do_sample=do_sample, top_k=topk, top_p=topp, length_penalty=lp, max_length=max_len, min_length=min_len, use_cache=use_cache)
        return result
