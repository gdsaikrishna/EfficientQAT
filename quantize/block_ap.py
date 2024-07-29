import torch
import torch.nn as nn
import torch.nn.functional as F
import quantize.int_linear_fake as int_linear_fake
import quantize.int_linear_real as int_linear_real

from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
import math
import utils
import pdb
import gc
from quantize.utils import (
    quant_parameters,weight_parameters,trainable_parameters,
    set_quant_state,quant_inplace,set_quant_parameters,
    set_weight_parameters,trainable_parameters_num,get_named_linears,set_op_by_name)
import time
from datautils_block import BlockTrainDataset
from torch.utils.data import DataLoader
import shutil
import os
def update_dataloader(layer, input_dataloader, dev, attention_mask, position_ids, output_dataloader = None):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for index1, inps in enumerate(input_dataloader):
                inps = inps.to(dev)
                # breakpoint()
                if isinstance(layer, int_linear_fake.QuantLinear) or isinstance(layer, nn.Linear):
                    inps = layer(inps).to('cpu')
                else:
                    inps = layer(inps, attention_mask=attention_mask,position_ids=position_ids)[0].to('cpu')
                batch_size = len(inps)
                for index2,inp in enumerate(inps):
                    
                    if output_dataloader is not None:
                        output_dataloader.dataset.update_data(index1*batch_size+index2,inp.clone())    # .clone() to avoid one saving bug of pytorch
                    else:
                        input_dataloader.dataset.update_data(index1*batch_size+index2,inp.clone())    # .clone() to avoid one saving bug of pytorch
                    

     
def block_ap(
    model,
    args,
    trainloader,
    valloader,
    logger=None,
):
    logger.info("Starting ...")
    if args.off_load_to_disk:
        logger.info("offload the training dataset to disk, saving CPU memory, but may slowdown the training due to additional I/O...")
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dev="cpu"
    use_cache = model.config.use_cache
    model.config.use_cache = False
    enable_lm_head_quant = args.lm_head
    # move embedding layer and first layer to target device
    # only suppress llama models now
    # Initialize the layers object and move initial layers to target device
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    is_llama = True     
    lm_head = model.lm_head
            
    # Moving first decoder block to target device
    if enable_lm_head_quant: lm_head = lm_head.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = torch.float16

    # path to save intermediate data
    flag = time.time()
    fp_train_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_train'
    fp_val_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_val'
    quant_train_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_train'
    quant_val_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_val'
    
    # if the above intermediate already exists remove the folders
    for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
        if os.path.exists(path):
            shutil.rmtree(path)

    # Get Float Train data of 4k samples and val data of 64 samples with seqlen 2048, 
    # These dataset will track the data for inputs of each layer
    hidden_size = 4096 if enable_lm_head_quant else model.config.hidden_size
    fp_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                hidden_size, dtype, cache_path=fp_train_cache_path,off_load_to_disk=args.off_load_to_disk)
    fp_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                hidden_size, dtype, cache_path=fp_val_cache_path,off_load_to_disk=args.off_load_to_disk)
    
    # Dataloader object loads the data from dataset
    fp_train_inps_loader = DataLoader(fp_train_inps, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,prefetch_factor=args.prefetch_factor)
    fp_val_inps_loader = DataLoader(fp_val_inps, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,prefetch_factor=args.prefetch_factor)
    
    if enable_lm_head_quant:
        fp_train_outs = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                151936, dtype, cache_path=fp_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        fp_val_outs = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                151936, dtype, cache_path=fp_val_cache_path,off_load_to_disk=args.off_load_to_disk)
        
        fp_train_outs_loader = DataLoader(fp_train_outs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,prefetch_factor=args.prefetch_factor)
        fp_val_outs_loader = DataLoader(fp_val_outs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,prefetch_factor=args.prefetch_factor)
    
    

    # catch the first layer input
    cache = {"i": 0}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            if cache["i"]<=args.train_size-1:
                fp_train_inps.update_data(cache["i"], inp.squeeze(0).to('cpu'))
            else:
                fp_val_inps.update_data(cache["i"]-args.train_size, inp.squeeze(0).to('cpu'))
            cache["i"] += 1
            if "attention_mask" in kwargs:
                cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama and "position_ids" in kwargs: 
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError
    if enable_lm_head_quant:
        model.model= model.model.to(dev)
        lm_head = Catcher(lm_head)
        # layers[0] = Catcher(layers[0])
    else:
        # Catcher module to catch layers[0] input
        layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama
    # Model inference to catch first decoder block input
    with torch.no_grad():
        for batch in (trainloader+valloader):
            if cache["i"] >= args.train_size + args.val_size:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    if enable_lm_head_quant:
        model.model = model.model.cpu()
    # model = model.to("cpu")
    
    # move embedding layer and first layer to cpu
    if not enable_lm_head_quant:
        layers[0] = layers[0].module
        layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    #Update the quant input data from float input data
    if args.off_load_to_disk:
        # copy quant input from fp input, they are same in first layer
        shutil.copytree(fp_train_cache_path, quant_train_cache_path)
        shutil.copytree(fp_val_cache_path, quant_val_cache_path)
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                    model.config.hidden_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk)
    else:
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                    model.config.hidden_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk)
        for index,data in enumerate(fp_train_inps):
            quant_train_inps.update_data(index, data)
        for index,data in enumerate(fp_val_inps):
            quant_val_inps.update_data(index, data)
            
    quant_train_inps_loader = DataLoader(quant_train_inps, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,prefetch_factor=args.prefetch_factor)
    quant_val_inps_loader = DataLoader(quant_val_inps, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,prefetch_factor=args.prefetch_factor)

    # Initialize the position_ids and attention_mask
    attention_mask = cache["attention_mask"] if "attention_mask" in cache else None
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    if is_llama and "position_ids" in cache:
        position_ids = cache["position_ids"]
    else:
        position_ids = None
    if enable_lm_head_quant:
        lm_head = lm_head.module
        logger.info(f"=== Start quantize blocks lm_head ===")
        lm_head = lm_head.to(dev)
        qlm_head = copy.deepcopy(lm_head)
        w_bits, group_size = 4, 64
        if isinstance(qlm_head, torch.nn.Linear):
            quantlinear = int_linear_fake.QuantLinear(lm_head, w_bits, args.group_size)
            qlm_head = quantlinear
        qlm_head.to(dev)
        set_quant_state(qlm_head, weight_quant=False)
        if args.epochs > 0:
            update_dataloader(qlm_head,fp_train_inps_loader,dev,attention_mask,position_ids,fp_train_outs_loader)
            update_dataloader(qlm_head,fp_val_inps_loader,dev,attention_mask,position_ids,fp_val_outs_loader)
        
        # activate quantization
        set_quant_state(qlm_head,weight_quant=True)  
        
        total_training_iteration = args.epochs * args.train_size / args.batch_size
        
        if args.epochs > 0:
            with torch.no_grad():
                qlm_head.float()      # fp32 is required for AMP training
            # create optimizer
            param = []
            assert args.quant_lr > 0 or args.weight_lr > 0
            param_group_index = 0
            # Initialize the quant params scale and zero point to train
            if args.quant_lr > 0:
                set_quant_parameters(qlm_head,True)
                param.append({"params":quant_parameters(qlm_head),"lr":args.quant_lr})
                empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)], lr=args.quant_lr)
                quant_scheduler = CosineAnnealingLR(empty_optimizer_1, T_max=total_training_iteration, eta_min=args.quant_lr/args.min_lr_factor)
                quant_index = param_group_index
                param_group_index += 1
            else:
                set_quant_parameters(qlm_head,False)
                
            # Initialize the weight params: weight, scale and zero point to train
            if args.weight_lr > 0:
                set_weight_parameters(qlm_head,True)
                param.append({"params":weight_parameters(qlm_head),"lr":args.weight_lr})
                empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)], lr=args.weight_lr)
                weight_scheduler = CosineAnnealingLR(empty_optimizer_2, T_max=total_training_iteration, eta_min=args.weight_lr/args.min_lr_factor)
                weight_index = param_group_index
                param_group_index += 1
            else:
                set_weight_parameters(qlm_head,False)
            
            #Initialize the optimizer
            optimizer = torch.optim.AdamW(param, weight_decay=args.wd)

            loss_scaler = utils.NativeScalerWithGradNormCount()
            trainable_number = trainable_parameters_num(qlm_head)
            print(f"trainable parameter number: {trainable_number/1e6}M")

            best_val_loss = 1e6
            early_stop_flag = 0
            from tqdm import tqdm
            for epoch in tqdm(range(args.epochs)):
                loss_list = []
                norm_list = []
                start_time = time.time()
                for index, (quant_inps, fp_inps) in enumerate(zip(quant_train_inps_loader, fp_train_outs_loader)): 
                    # print(index, end=" ")   
                    # obtain output of quantization model
                    with torch.cuda.amp.autocast():
                        input = quant_inps.to(dev)
                        label = fp_inps.to(dev)
                        quant_out = qlm_head(input)[0]
                        reconstruction_loss = loss_func(label, quant_out)
                        loss =  reconstruction_loss
                    del input, label, quant_out

                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                    loss_list.append(reconstruction_loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters=trainable_parameters(qlm_head)).cpu()
                    norm_list.append(norm.data)

                    # adjust lr
                    if args.quant_lr > 0:
                        quant_scheduler.step()
                        optimizer.param_groups[quant_index]['lr'] = quant_scheduler.get_lr()[0]
                    if args.weight_lr >0 :
                        weight_scheduler.step()
                        optimizer.param_groups[weight_index]['lr'] = weight_scheduler.get_lr()[0]

                val_loss_list = []
                for index, (quant_inps,fp_inps) in enumerate(zip(quant_val_inps_loader, fp_train_outs_loader)):  
                    # obtain output of quantization model
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            input = quant_inps.to(dev)
                            label = fp_inps.to(dev)
                            quant_out = qlm_head(input)[0]
                            reconstruction_loss = loss_func(label, quant_out)
                    val_loss_list.append(reconstruction_loss.cpu())
                 
                train_mean_num = min(len(loss_list),64) # calculate the average training loss of last train_mean_num samples
                loss_mean = torch.stack(loss_list)[-(train_mean_num-1):].mean()
                val_loss_mean = torch.stack(val_loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"block lm_head epoch {epoch} recon_loss:{loss_mean} val_loss:{val_loss_mean} quant_lr:{quant_scheduler.get_lr()[0]} norm:{norm_mean:.8f} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2} time {time.time()-start_time} ")
                if val_loss_mean < best_val_loss:
                    best_val_loss = val_loss_mean
                else:
                    early_stop_flag += 1
                    if args.early_stop > 0 and early_stop_flag >=args.early_stop:
                        break
            optimizer.zero_grad()
            del optimizer

        # real smooth and quantization
        qlm_head.half()
        # directly replace the weight with fake quantization
        quant_inplace(qlm_head)
        set_quant_state(qlm_head,weight_quant=False)  # weight has been quantized inplace
        # if args.epochs>0:
        #     # update inputs of quantization model
        #     update_dataloader(qlm_head,quant_train_inps_loader,dev,attention_mask,position_ids)
        #     update_dataloader(qlm_head,quant_val_inps_loader,dev,attention_mask,position_ids)
        
        # move to cpu
        lm_head = qlm_head.to("cpu")

        # pack quantized weights, note that this process is slow on poor CPU or busy CPU
        if args.real_quant:
            named_linears = get_named_linears(qlm_head, int_linear_fake.QuantLinear)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scale.clamp(1e-4,1e4).detach()
                zeros = module.weight_quantizer.zero_point.detach().cuda().round().cpu()
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1).transpose(0,1).contiguous()
                zeros = zeros.view(dim0,-1).transpose(0,1).contiguous()
                q_linear = int_linear_real.QuantLinear(w_bits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                # set_op_by_name(qlayer, name, q_linear)       
                logger.info(f"pack quantized {name} finished")
                del module        
        del lm_head
        torch.cuda.empty_cache()
    
    else:
    # Iterating through each decoder block
        for block_index in range(len(layers)):
            logger.info(f"=== Start quantize blocks {block_index}===")
            #Move the decoder block to the target device
            layer = layers[block_index].to(dev)
            qlayer = copy.deepcopy(layer)
            # replace torch.nn.Linear with QuantLinear for QAT
            w_bits = args.wbits
            if block_index in [22,23,24,25,26,27]:
                w_bits = 2
            for name, module in qlayer.named_modules():
                if isinstance(module,torch.nn.Linear):
                    quantlinear = int_linear_fake.QuantLinear(module, w_bits, args.group_size)
                    set_op_by_name(qlayer, name, quantlinear)  
                    del module  
            
            
            # obtain output of full-precision model
            # Now move the qlayer to the target device
            qlayer.to(dev)
            # deactivate quantization for obtaining ground truth
            set_quant_state(qlayer,weight_quant=False)
            if args.epochs > 0:
                update_dataloader(qlayer,fp_train_inps_loader,dev,attention_mask,position_ids)
                update_dataloader(qlayer,fp_val_inps_loader,dev,attention_mask,position_ids)
            
            # activate quantization
            set_quant_state(qlayer,weight_quant=True)  
            
            total_training_iteration = args.epochs * args.train_size / args.batch_size
            
            if args.epochs > 0:
                with torch.no_grad():
                    qlayer.float()      # fp32 is required for AMP training
                # create optimizer
                param = []
                assert args.quant_lr > 0 or args.weight_lr > 0
                param_group_index = 0
                # Initialize the quant params scale and zero point to train
                if args.quant_lr > 0:
                    set_quant_parameters(qlayer,True)
                    param.append({"params":quant_parameters(qlayer),"lr":args.quant_lr})
                    empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)], lr=args.quant_lr)
                    quant_scheduler = CosineAnnealingLR(empty_optimizer_1, T_max=total_training_iteration, eta_min=args.quant_lr/args.min_lr_factor)
                    quant_index = param_group_index
                    param_group_index += 1
                else:
                    set_quant_parameters(qlayer,False)
                    
                # Initialize the weight params: weight, scale and zero point to train
                if args.weight_lr > 0:
                    set_weight_parameters(qlayer,True)
                    param.append({"params":weight_parameters(qlayer),"lr":args.weight_lr})
                    empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)], lr=args.weight_lr)
                    weight_scheduler = CosineAnnealingLR(empty_optimizer_2, T_max=total_training_iteration, eta_min=args.weight_lr/args.min_lr_factor)
                    weight_index = param_group_index
                    param_group_index += 1
                else:
                    set_weight_parameters(qlayer,False)
                
                #Initialize the optimizer
                optimizer = torch.optim.AdamW(param, weight_decay=args.wd)

                loss_scaler = utils.NativeScalerWithGradNormCount()
                trainable_number = trainable_parameters_num(qlayer)
                print(f"trainable parameter number: {trainable_number/1e6}M")

                best_val_loss = 1e6
                early_stop_flag = 0
                from tqdm import tqdm
                for epoch in tqdm(range(args.epochs)):
                    loss_list = []
                    norm_list = []
                    start_time = time.time()
                    for index, (quant_inps, fp_inps) in enumerate(zip(quant_train_inps_loader, fp_train_inps_loader)):    
                        # obtain output of quantization model
                        with torch.cuda.amp.autocast():
                            input = quant_inps.to(dev)
                            label = fp_inps.to(dev)
                            quant_out = qlayer(input, attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                            reconstruction_loss = loss_func(label, quant_out)
                            loss =  reconstruction_loss

                        if not math.isfinite(loss.item()):
                            logger.info("Loss is NAN, stopping training")
                            pdb.set_trace()
                        loss_list.append(reconstruction_loss.detach().cpu())
                        optimizer.zero_grad()
                        norm = loss_scaler(loss, optimizer,parameters=trainable_parameters(qlayer)).cpu()
                        norm_list.append(norm.data)

                        # adjust lr
                        if args.quant_lr > 0:
                            quant_scheduler.step()
                            optimizer.param_groups[quant_index]['lr'] = quant_scheduler.get_lr()[0]
                        if args.weight_lr >0 :
                            weight_scheduler.step()
                            optimizer.param_groups[weight_index]['lr'] = weight_scheduler.get_lr()[0]

                    val_loss_list = []
                    for index, (quant_inps,fp_inps) in enumerate(zip(quant_val_inps_loader, fp_val_inps_loader)):  
                        # obtain output of quantization model
                        with torch.no_grad():
                            with torch.cuda.amp.autocast():
                                input = quant_inps.to(dev)
                                label = fp_inps.to(dev)
                                quant_out = qlayer(input, attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                                reconstruction_loss = loss_func(label, quant_out)
                        val_loss_list.append(reconstruction_loss.cpu())
                    
                    train_mean_num = min(len(loss_list),64) # calculate the average training loss of last train_mean_num samples
                    loss_mean = torch.stack(loss_list)[-(train_mean_num-1):].mean()
                    val_loss_mean = torch.stack(val_loss_list).mean()
                    norm_mean = torch.stack(norm_list).mean()
                    logger.info(f"blocks {block_index} epoch {epoch} recon_loss:{loss_mean} val_loss:{val_loss_mean} quant_lr:{quant_scheduler.get_lr()[0]} norm:{norm_mean:.8f} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2} time {time.time()-start_time} ")
                    if val_loss_mean < best_val_loss:
                        best_val_loss = val_loss_mean
                    else:
                        early_stop_flag += 1
                        if args.early_stop > 0 and early_stop_flag >=args.early_stop:
                            break
                optimizer.zero_grad()
                del optimizer

            # real smooth and quantization
            qlayer.half()
            # directly replace the weight with fake quantization
            quant_inplace(qlayer)
            set_quant_state(qlayer,weight_quant=False)  # weight has been quantized inplace
            if args.epochs>0:
                # update inputs of quantization model
                update_dataloader(qlayer,quant_train_inps_loader,dev,attention_mask,position_ids)
                update_dataloader(qlayer,quant_val_inps_loader,dev,attention_mask,position_ids)
            
            # move to cpu
            layers[block_index] = qlayer.to("cpu")

            # pack quantized weights, note that this process is slow on poor CPU or busy CPU
            if args.real_quant:
                named_linears = get_named_linears(qlayer, int_linear_fake.QuantLinear)
                for name, module in named_linears.items():
                    scales = module.weight_quantizer.scale.clamp(1e-4,1e4).detach()
                    zeros = module.weight_quantizer.zero_point.detach().cuda().round().cpu()
                    group_size = module.weight_quantizer.group_size
                    dim0 = module.weight.shape[0]
                    scales = scales.view(dim0,-1).transpose(0,1).contiguous()
                    zeros = zeros.view(dim0,-1).transpose(0,1).contiguous()
                    q_linear = int_linear_real.QuantLinear(w_bits, group_size, module.in_features,module.out_features,not module.bias is None)
                    q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                    set_op_by_name(qlayer, name, q_linear)       
                    logger.info(f"pack quantized {name} finished")
                    del module        
            del layer
            torch.cuda.empty_cache()

    # delete cached dataset
    for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
        if os.path.exists(path):
            shutil.rmtree(path)

    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

