from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import math
import random
import numpy
import os
import json
from utils.utils import (
    load_tokenizer, MixedPrecisionManager, load_pickle, 
    EarlyStopController,  LabelSmoother, class_Mul_Vul_IterableDataset,
    prepare_vulclassify_dataset
)

from utils.metrics import compute_acc, compute_p_r_f1

from transformers import (get_scheduler, DataCollatorForLanguageModeling)
from models.pretrain_model.pretrain_module_welk import WelkirForVulclassifyModel
from models.pretrain_model.run_pretrain_module import load_checkpoint_if_exists, get_data_paths

from prettytable import PrettyTable
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


# add TSNE
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def VulclassifyModel_Save(model, optimizer, lr_scheduler, scaler, config, save_dir, epoch):
    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "model_state_dict": model_to_save.state_dict(),            
        "optimizer_state_dict": optimizer.state_dict(),           
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),      
        "epoch": epoch,    
        "config": config,  
    }
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    save_path = os.path.join(save_dir, "vulclassify_model_checkpoint.pt")
    torch.save(checkpoint, save_path)
    logger.info(f"Model, optimizer, and scheduler checkpoint for epoch {epoch} saved to {save_path}")


def human_format(num):
    """Transfer number into a readable format."""
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


def layer_wise_parameters(model):
    """Returns a printable table representing the layer-wise model parameters, their shapes and numbers"""
    table = PrettyTable()
    table.field_names = ["Layer Name", "Output Shape", "Param #"]
    table.align["Layer Name"] = "l"
    table.align["Output Shape"] = "r"
    table.align["Param #"] = "r"
    for name, parameters in model.named_parameters():
        if parameters.requires_grad:
            table.add_row([name, str(list(parameters.shape)), parameters.numel()])
    return table


def model_summary(model):
    if hasattr(model, "config"):
        logger.debug(f"Model configuration:\n{model.config}")
    logger.info(f"Model type: {model.__class__.__name__}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {human_format(num_params)}")
    logger.debug(f"Layer-wise parameters:\n{layer_wise_parameters(model)}")



def run_vulclassify_task(config):
    logger.info(f"run_vulclassify_task, start Loading tokenizer ")
    tokenizer, cfg_vocab, dfg_vocab, rdg_vocab = load_tokenizer(config)
   
    if config.only_prepare_vulclassify_dataset:
        logger.info(f"start prepare_vulclassify_dataset...")
        prepare_vulclassify_dataset(config=config, tokenizer=tokenizer, cfg_vocab=cfg_vocab, 
                                  dfg_vocab=dfg_vocab, rdg_vocab=rdg_vocab, split=config.dataset_split)
        logger.info(f"finish prepare_vulclassify_dataset...")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"*****Loading model, device:{device}*****")
    assert os.path.exists(config.WelkirModel_path)
    logger.info(f"WelkirModel_path: {config.WelkirModel_path}")
    
    train_module = WelkirForVulclassifyModel.from_pretrained(config.WelkirModel_path)
    # train_module = torch.nn.DataParallel(train_module, device_ids=[0, 1])
    train_module = torch.nn.DataParallel(train_module, device_ids=[0,1,2,3,4,5,6,7])
    # model_summary(model)
    model = train_module.to(device)
    
    # --------------------------------------------------
    # Train and valid
    # --------------------------------------------------

    model, epoch = vulclassify_train(
            config, model=model, tokenizer=tokenizer, cfg_vocab=cfg_vocab,
            dfg_vocab=dfg_vocab, rdg_vocab=rdg_vocab, device=device, vulclassify_only_test=config.only_test
        )
    torch.cuda.empty_cache()

    logger.info(f"config.only_test:{config.only_test}, get model epoch: {epoch}")
    
    # --------------------------------------------------
    # validation  代码片段的测试结果
    # --------------------------------------------------
    
    valid_results = vulclassify_validation(
        config, model=model, tokenizer=tokenizer, cfg_vocab=cfg_vocab, 
            dfg_vocab=dfg_vocab, rdg_vocab=rdg_vocab, device=device
    )
    logger.info(f"vulclassify_validation finish, valid_results: {valid_results}")
    

    test_results = vulclassify_test(
        config, model=model, tokenizer=tokenizer, cfg_vocab=cfg_vocab, 
            dfg_vocab=dfg_vocab, rdg_vocab=rdg_vocab, device=device
    )
    logger.info(f"vulclassify_test finish, test_results: {test_results}")




def create_Vul_dataloader(config, dataset_root, shuffle=False, pin_memory=True):

    Listfiles_paths, total_samples  = get_data_paths(dataset_root, ".pk")
    num_workers = config.DataLoader_num_workers
    logger.info(f"Listfiles Count: {len(Listfiles_paths)}, total_samples: {total_samples}")

    dataset = class_Mul_Vul_IterableDataset(Listfiles_paths)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
       
    return dataloader, total_samples


def vulclassify_train(config, model, tokenizer, cfg_vocab, dfg_vocab, rdg_vocab, device, vulclassify_only_test=None):
    logger.info("*****vulclassify_train Loading train create_dataloader *****")
    
    dataset_dir = config.dataset_root
    if not vulclassify_only_test:
        train_dataset_path = os.path.join(dataset_dir, "train")
        train_dataloader, Len_train_dataloader = create_Vul_dataloader(config, train_dataset_path)

        logger.info("*****************Start validation create_dataloader*****************")
        validation_dataset_path = os.path.join(dataset_dir, "validation")
        validation_dataloader, Len_validation_dataloader = create_Vul_dataloader(config, validation_dataset_path)

    else:
        logger.info("*****************Start vulclassify_only_test :{vulclassify_only_test}*****************")
        Len_train_dataloader = 0
        Len_validation_dataloader = 0



    logger.info("***** Preparing Training Utils *****")
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=float(config.learning_rate)
    )
    # calculate max steps
    num_update_steps_per_epoch = math.ceil(
        Len_train_dataloader / config.gradient_accumulation_steps
    )
    if config.max_train_steps is None:
        config.max_train_steps = config.num_epochs * num_update_steps_per_epoch
    if config.warmup_steps >= 1:
        config.warmup_steps = int(config.warmup_steps)
    elif config.warmup_steps >= 0:
        config.warmup_steps = int(config.warmup_steps * config.max_train_steps)
    else:
        raise ValueError(f"Invalid warmup steps: {config.warmup_steps}")
    
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_train_steps,
    )

    # mixed precision
    amp = MixedPrecisionManager(activated=True) 

    # early stop
    early_stop = EarlyStopController(
        patience=config.patience,
        best_model_dir=os.path.join(config.VulclassifyModel_path, f"best_acc"),
        higher_is_better=True,
    )

    # label smoothing
    label_smoother = None
    if config.label_smoothing_factor != 0:
        label_smoother = LabelSmoother(epsilon=config.label_smoothing_factor)


    # batch size per device, total batch size
    if config.num_devices > 1:
        batch_size_per_device = config.batch_size // config.num_devices
        if batch_size_per_device * config.num_devices != config.batch_size:
            raise ValueError(
                f"The total batch size {config.batch_size=} is not an integer multiple "
                f"of the device count: {config.num_devices}"
            )
    else:
        batch_size_per_device = config.batch_size
    total_batch_size = config.batch_size * config.gradient_accumulation_steps



    logger.info("***** Training *****")
    logger.info(f"  Num examples = {Len_train_dataloader}")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Batch size per device = {batch_size_per_device}")
    logger.info(
        f"  Total train batch size (w. parallel & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}"
    )

    completed_steps = 0     # optimization step
    model.zero_grad()
    checkpoint_path = os.path.join(config.VulclassifyModel_path, "vulclassify_model_checkpoint.pt")
    start_epoch =  load_checkpoint_if_exists(model, optimizer, lr_scheduler, amp.scaler if amp.activated else None, checkpoint_path)

    if vulclassify_only_test:
        return model, start_epoch

    epoch = start_epoch
    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        epoch_loss = 0
        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}", ascii=True)
        for step, batch in enumerate(epoch_iterator):
            optimizer.zero_grad()
            batch.pop('base_filename', None)
            with amp.context():
                # places all tensors in the dict to the right device
                for k, v in batch.items():
                    batch[k] = v.to(device)
              
                # when using label smoothing, we calculate loss by ourselves
                if label_smoother is not None:
                    labels = batch.pop("labels")
                else:
                    labels = None
                    
                # passes batch through model
                outputs, pooled_output = model(
                    input_ids=batch['input_ids'],
                    all_inst_encoder_input_ids=batch['all_inst_encoder_input_ids'],
                    total_group_count=batch['total_group_count'],
                    input_inst_tokencount=batch['input_inst_tokencount'],
                    input_instcount_list=batch['input_instcount_list'],
                    cfg_matrix=batch['cfg_matrix'],
                    dfg_matrix=batch['dfg_matrix'],
                    rdg_matrix=batch['rdg_matrix'],
                    labels=batch['labels'],
                    output_attentions=True,
                    return_dict=True)

                
                # calculates or gets the loss
                if label_smoother is not None:
                    loss = label_smoother(outputs, labels)
                else:
                    loss = outputs.loss

                # gets mean loss
                if config.num_devices > 1:
                    loss = loss.mean()
                # normalizes loss
                loss = loss / config.gradient_accumulation_steps

            # backwards loss by amp manager
            amp.backward(loss)
            epoch_loss += loss.item()

            if (
                    (step + 1) % config.gradient_accumulation_steps == 0
                    or step == Len_train_dataloader - 1
            ):
                # amp.step() includes gradient clipping and optimizer.step()
                logger.info(f"step:{step}, Before optimizer.step(): Learning rate: {optimizer.param_groups[0]['lr']}")
                amp.step(model, optimizer, max_grad_norm=config.max_grad_norm)
                lr_scheduler.step()
                logger.info(f"step:{step}, After lr_scheduler.step(): Learning rate: {optimizer.param_groups[0]['lr']}")


            labels = batch['labels']
            logger.info(f"Epoch {epoch}, Step {step}, loss:{loss}, epoch_loss: {epoch_loss}")
            
 
        logger.info(f"Epoch {epoch+1} finished. ")
        


        epoch_save_dir_path = os.path.join(config.VulclassifyModel_path, str(epoch))
        VulclassifyModel_Save( model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    scaler=amp.scaler if amp.activated else None,
                    config=config,
                    save_dir=epoch_save_dir_path,
                    epoch=epoch+1,
                )
        

        VulclassifyModel_Save( model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    scaler=amp.scaler if amp.activated else None,
                    config=config,
                    save_dir=config.VulclassifyModel_path,
                    epoch=epoch+1,
                )

        
        logger.info("*****************Start validation*****************")
        valid_results = __vulclassify_eval(
            config=config,
            model=model,
            dataloader=validation_dataloader,
            Len_dataloader=Len_validation_dataloader,
            split="valid",
            epoch=epoch,
            device=device
        )

        early_stop(
            score=valid_results[f"valid/{config.metric_for_best_model}"],
            model=model,
            epoch=epoch,
            step=completed_steps,
        )
        if not early_stop.hit:
            logger.info(
                f"Early stopping counter: {early_stop.counter}/{early_stop.patience}"
            )

        if early_stop.early_stop:
            logger.info(f"Early stopping is triggered")
            break

    logger.info("End of training")
    # load best model at end of training
    model = early_stop.load_best_model()
    return model, epoch


def __vulclassify_eval(config, model, dataloader, Len_dataloader, split, device, epoch=None):
    
    logger.info(f"__vulclassify_eval, split:{split}, epoch:{epoch} ")
    
    assert split in ["valid", "test"]
    # assert split == "test" or epoch is not None

    # statistics
    num_examples = 0
    num_steps = 0
    all_loss = []

    # used for computing metrics
    all_golds = []
    all_preds = []
    all_filenames = []

    # TSNE
    all_pooled_outputs = []
    
    total_detect = Len_dataloader/config.num_devices
    eval_bar = tqdm(dataloader, total=Len_dataloader, ascii=True)
    model.eval()
    for step, batch in enumerate(eval_bar):
        filenames = batch.pop('base_filename', None)
        if filenames is not None:
            all_filenames.extend(filenames)

        for k, v in batch.items():
            batch[k] = v.to(device)
        labels = batch.get("labels")

        with torch.no_grad():
            # outputs = model(**batch)
            outputs, outputs_pooled_outputs = model(
                input_ids=batch['input_ids'],
                all_inst_encoder_input_ids=batch['all_inst_encoder_input_ids'],
                total_group_count=batch['total_group_count'],
                input_inst_tokencount=batch['input_inst_tokencount'],
                input_instcount_list=batch['input_instcount_list'],
                cfg_matrix=batch['cfg_matrix'],
                dfg_matrix=batch['dfg_matrix'],
                rdg_matrix=batch['rdg_matrix'],
                labels=batch['labels'],
                output_attentions=True,
                return_dict=True)

            loss = outputs.loss
            all_loss.append(loss.mean().item())

            logits = outputs.logits
            preds = numpy.argmax(logits.cpu().numpy(), axis=1)
            all_preds.extend([p.item() for p in preds])
            all_golds.extend(labels.squeeze(-1).cpu().numpy().tolist())

            # add: Collect pooled output for t-SNE and centroid distance
            # pooled_output = outputs_pooled_outputs.cpu().numpy()
            # all_pooled_outputs.append(pooled_output)
            
            
        num_examples += len(labels)
        num_steps += 1

    eval_loss = numpy.mean(all_loss)

    results = compute_acc(preds=all_preds, golds=all_golds, prefix=split)
    if config.num_labels == 2:
        results.update(compute_p_r_f1(preds=all_preds, golds=all_golds, prefix=split))
    elif config.num_labels > 2:
        results.update(compute_p_r_f1(preds=all_preds, golds=all_golds, prefix=split, average='macro'))

    results.update({f"{split}/loss": eval_loss})
    logger.info(results)

    results.update({
        f"{split}/num_examples": num_examples,
        f"{split}/num_steps": num_steps,
    })
    logger.info(results)

    # make sure that the metric for selecting best model is in the results when validating
    if split == "valid" and f"{split}/{config.metric_for_best_model}" not in results:
        raise ValueError(
            f"The metric for selecting best model is set to {config.metric_for_best_model}, "
            f"which is, however, not found in to validation results."
        )

    logger.info(f"Start gathering and saving {split} results and details...")
    # save results to json file
    save_dir = os.path.join(
        config.save_eval_dir, f"valid_epoch_{epoch}" if split == "valid" else "test"
    )

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "results.json"), mode="w", encoding="utf-8") as f:
        json.dump(results, f)


    detailed_results = []
    for filename, pred, gold in zip(all_filenames, all_preds, all_golds):
        detailed_results.append({
            "filename": filename,
            "prediction": pred,
            "gold_label": gold
        })

    with open(os.path.join(save_dir, "detailed_results.json"), mode="w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=4)

    logger.info(f"{split.capitalize()} results and details are saved to {save_dir}")
    model.train()

    ###########################################t-SNE########################################################
    # all_pooled_outputs = numpy.vstack(all_pooled_outputs)
    # logger.info(f"all_pooled_outputs.shape: {all_pooled_outputs.shape}")
    # all_golds = numpy.array(all_golds)
    
    # custom_colors = ['#FFA500', '#1F77B4', '#2CA02C', '#D62728', '#9467BD']
    # cmap = ListedColormap(custom_colors)
    # tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    # pooled_outputs_2d = tsne.fit_transform(all_pooled_outputs)
    # plt.figure(figsize=(12, 8))
    # scatter = plt.scatter(
    #     pooled_outputs_2d[:, 0],
    #     pooled_outputs_2d[:, 1],
    #     c=all_golds,
    #     cmap=cmap,
    #     s=20,
    #     alpha=1
    # )

    # plt.title("WelkIR", pad=20)
    # plt.xticks([])
    # plt.yticks([])
    # plt.tick_params(axis='both', which='both', bottom=False, left=False)
    # save_path = os.path.join(save_dir, "WelkIR_t-SNE_5class.png")
    # logger.info(f"save_path: {save_path}")
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.close()
    
    ###########################################t-SNE########################################################
    return results

def vulclassify_validation(config, model, tokenizer, cfg_vocab, dfg_vocab, rdg_vocab, device):
    logger.info("***** vulclassify_validation *****")
    dataset_dir = config.dataset_root
    validation_dataset_path = os.path.join(dataset_dir, "validation")
    validation_dataloader, Len_validation_dataloader = create_Vul_dataloader(config, validation_dataset_path)
        
    logger.info(f"Test dataset is prepared, size: {Len_validation_dataloader}")

    model.to(device)
    if config.num_devices > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    validation_results = __vulclassify_eval(
        config=config,
        model=model,
        dataloader=validation_dataloader,
        Len_dataloader=Len_validation_dataloader,
        split="valid",
        device=device
    )
    logger.info("End vulclassify_validation of testing")
    return validation_results


def vulclassify_test(config, model, tokenizer, cfg_vocab, dfg_vocab, rdg_vocab, device):
    logger.info("*****vulclassify Testing *****")
    dataset_dir = config.dataset_root
    test_dataset_path = os.path.join(dataset_dir, "test")
    ######################## debug################################
    # test_dataset_path = os.path.join(dataset_dir, "validation")
    test_dataloader, Len_test_dataloader = create_Vul_dataloader(config, test_dataset_path)

    logger.info(f"Test dataset is prepared, size: {Len_test_dataloader}")

    model.to(device)
    if config.num_devices > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    test_results = __vulclassify_eval(
        config=config,
        model=model,
        dataloader=test_dataloader,
        Len_dataloader=Len_test_dataloader,
        split="test",
        device=device
    )
    logger.info("End vulclassify_test of testing")
    return test_results
