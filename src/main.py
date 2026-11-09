import logging
import numpy as np
import torch
import warnings
from tqdm import tqdm
import os
import pandas as pd
from pathlib import Path

from spaformer.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    set_imitate_seed,
    Logger,
    get_current_lr,
    load_best_configs,
    load_dataset,
    iter_cell_batches,
    encoding_mask,
    encoding_mask_nonzeros,
    objective_metadata
)
import sys

from spaformer.models import build_model


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def save_cluster(sample_number, ari, ari_refine, nmi, nmi_refine, num_layers, max_epoch, output_dir="results/run"):
    save_path = output_dir
    
    
    metrics_dict = {
                    "slice_number": str(sample_number),
                    "ari": f"{ari}",
                    "ari_refine": f"{ari_refine}", 
                    "nmi": f"{nmi}",
                    "nmi_refine": f"{nmi_refine}"
                    }
       
       
    outdir = output_dir
    Path(outdir).mkdir(parents=True, exist_ok=True)

    csv_path = Path(os.path.join(outdir, "ADEPT_cluster_result.csv"))
    df = pd.DataFrame([metrics_dict])
    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

def save_cluster_avg(sample_number, ari_mean, ari_std, nmi_mean, nmi_std, num_layers, max_epoch, output_dir="results/run"):
    save_path = output_dir
    
    metrics_dict = {
                    "slice_number": str(sample_number),
                    "ari_mean": f"{ari_mean:.3f}", 
                    "ari_std": f"{ari_std:.4f}",  
                    "nmi_mean": f"{nmi_mean:.3f}", 
                    "nmi_std": f"{nmi_std:.4f}"
                    }
          
    outdir = output_dir
    Path(outdir).mkdir(parents=True, exist_ok=True)

    csv_path = Path(os.path.join(outdir, "average_cluster_std.csv"))
    df = pd.DataFrame([metrics_dict])
    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
    



def save_imputation(sample_number, l1, cos_sim, rmse, num_layers, max_epoch, output_dir="results/run"):
    save_path = output_dir
    metrics_dict = {
                    "slice_number": str(sample_number),
                    "l1": f"{l1}", 
                    "cos_sim": f"{cos_sim}",
                    "rmse": f"{rmse}"
                    }
       
       
    outdir = output_dir
    Path(outdir).mkdir(parents=True, exist_ok=True)

    csv_path = Path(os.path.join(outdir, "imputation_result.csv"))
    df = pd.DataFrame([metrics_dict])
    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


    
def save_imputation_avg(sample_number, l1s, cos_sims, rmses, num_layers, max_epoch, output_dir="results/run"):
    save_path = output_dir
    
    mean_l1s = np.mean(l1s)
    std_l1s = np.std(l1s)
    mean_cos = np.mean(cos_sims)
    std_cos = np.std(cos_sims)
    mean_rmses = np.mean(rmses)
    std_rmses = np.std(rmses)
    
    metrics_dict = {
                    "slice_number": str(sample_number),
                    "mean_l1s": f"{mean_l1s:.3f}", 
                    "std_l1s": f"{std_l1s:.4f}", 
                    "mean_cos": f"{mean_cos:.3f}", 
                    "std_cos":  f"{std_cos:.4f}", 
                    "mean_rmses": f"{mean_rmses:.3f}", 
                    "std_rmses": f"{std_rmses:.4f}"
                    }
       
    outdir = output_dir
    Path(outdir).mkdir(parents=True, exist_ok=True)

    csv_path = Path(os.path.join(outdir, "imputation_result_std.csv"))
    df = pd.DataFrame([metrics_dict])
    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)





def pretrain(args, model, graphs, optimizer, max_epoch, device, scheduler, num_classes, loginfo, logger=None, mask=None, test_all=False, test_pe=False):
    test_id = -1
    logging.info("start training..")
    epoch_iter = tqdm(range(max_epoch))
    graphs = [g.to(device) for g in graphs]
    val_list = []
    batch_size = args.batch_size 
    mode = args.mode
    mask_ratio = args.mask_ratio
    anchor_seed = 42    
    for epoch in epoch_iter:
        count = 0
        model.train()
        
        
        for i in range(len(graphs)):
            graph = graphs[i]
            
            '''Processed_X with masked'''
            # X = graph.ndata['input']
            X = graph.ndata['gt']
            X_bins = graph.ndata['bin_val']
            z_val = graph.ndata['z_val']
            adj = graph.ndata['adj']
            X_pca = graph.ndata['X_pca']
            
            if args.task == "imputation":
                use_x, mask_in, mask_train = encoding_mask_nonzeros(X, mask, mask_ratio)
            elif args.task == "cluster":
                mask = torch.zeros_like(X, dtype=torch.bool)
                mask_in, mask_train = encoding_mask(X, mask_ratio)
            epoch_generator = (torch.Generator(device=device).manual_seed(anchor_seed + epoch) if args.replicate_exp else None)
            
            for target_idx, cell_ids_cpu in iter_cell_batches(adjacency_full=adj, batch_size=batch_size, mode=mode, shuffle=True, generator=epoch_generator, device=device):
                loss, cache_payload = model(args, graph, X, X_pca, X_bins, z_val, mask_in, mask_train, cell_ids_cpu,target_idx, epoch, count)
                count += 1
                if loss is None:
                    print("loss is None")
                    continue
                
                
                print("loss: ", loss)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                optimizer.step()
                
                # if cache_payload:
                #     model.update_negative_cache(cache_payload)
            
        if scheduler is not None:
            scheduler.step()

        # epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
    return model



    
def main(args):
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    import json
    (Path(args.output_dir) / "config.json").write_text(json.dumps(vars(args), indent=2))
    # seed = args.seed
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    num_layers = args.num_layers
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    
    optim_type = args.optimizer
    
    lr = args.lr
    weight_decay = args.weight_decay
    load_model = args.load_model
    use_scheduler = args.scheduler
    standardscale = args.standardscale
    if args.pe == 'None': args.pe = None
    pe = str(args.pe)
    objective = args.objective
    data_path = args.data_path
    pc = args.noise_pc
    test_pe = args.test_pe
    cache_path = args.cache_path
    
    metadata = {
    "task": args.task,
    **objective_metadata(args),
    }
    
    if args.task == "cluster":
        seed = 1
        # set_imitate_seed(seed)
    elif args.task == "imputation":
        seed = 0
        # np.random.seed(seed)
    
    for sample_number in [args.sample]:
        # sample_number = str(sample_number)
        
        if args.task == "cluster": 
            test_aris = []
            test_nmis = []
            # total = 20
            total = args.repetitions
        elif args.task == "imputation":
            test_l1s = []
            test_cos_sims = []
            test_rmses = []
            total = args.repetitions
            # total = 1
        
        for repeat_id in range(total):
            split_version = args.imputation_split[repeat_id] if args.task == "imputation" else repeat_id
            graphs, num_cells, num_features, num_classes, eval_mask, label = load_dataset(args, dataset_name, sample_number, pc, split_version, standardscale, data_path, cache_path=cache_path)
            '''Here we only let eval_mask to be single mask'''
            eval_mask = eval_mask[0]
            eval_mask = eval_mask.to(device)
            if args.task == "imputation":
                
                set_random_seed(split_version)
            elif args.task == "cluster":
                set_random_seed(split_version)
            
            args.num_cells = num_cells
            args.num_features = num_features
            

            acc_list = []
            estp_acc_list = []
            loginfo = None
            logger = None

            model = build_model(args)
            model.to(device)
            
            if args.first_time:
                
                torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "seed": seed,
                },
                str(Path(args.checkpoint_dir) / f"{sample_number}_shared_initial_state_{args.num_layers}_layers_test.pt"),                    
                )
                sys.exit()
            
            if args.replicate_exp:
                checkpoint = torch.load(
                    str(Path(args.checkpoint_dir) / f"{sample_number}_shared_initial_state_{args.num_layers}_layers_test.pt"),
                    map_location = torch.device(device),
                    weights_only=True
                )
                
                model.load_state_dict(
                    checkpoint["model_state_dict"]
                )
                       
            optimizer = create_optimizer(optim_type, model, lr, weight_decay)

            if use_scheduler:
                logging.info("Use schedular")
                scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
                # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                        # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
            else:
                scheduler = None

            if not load_model:
                torch.set_num_threads(1) 
                model = pretrain(args, model, graphs, optimizer, max_epoch, device, scheduler, num_classes, loginfo, logger, eval_mask, test_pe=test_pe)
                torch.set_num_threads(8) 
                
                model = model.cpu()

            if load_model:
                logging.info("Loading Model ... ")
                model.load_state_dict(torch.load(args.model_path, map_location=torch.device(device)))

            model = model.to(device)
            model.eval()
            run_name = None
            with torch.no_grad():
                if args.replicate_exp:
                    
                    run_name = (
                        f"{args.task}_{metadata['objective']}"
                        f"_cls{int(metadata['use_cls'])}"
                        f"_repeat{split_version}"
                    )

                    checkpoint_dir = Path(args.checkpoint_dir) / sample_number / run_name
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)

                    payload = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "num_epochs": args.max_epoch,
                        "repeat_id": split_version,
                        **metadata,
                    }

                    checkpoint_path = checkpoint_dir / (
                        f"{sample_number}_{args.num_layers}layers_"
                        f"{args.max_epoch}epochs_model.pt"
                    )

                    torch.save(payload, checkpoint_path)
                    
                    '''create checkpoint_path at directory you want to save model checkpoint'''
                    
                    # checkpoint = torch.load(
                    #     checkpoint_path,
                    #     map_location=f"cuda:{device}",
                    #     weights_only=False,
                    # )
                    
                    # model.load_state_dict(checkpoint["model_state_dict"])
                    # model.to(device)            
                
                eval_mask = eval_mask.float()
                
                if args.task == "cluster":
                    test_ari, test_ari_ref, test_nmi, test_nmi_ref = model.batch_test(args, graphs, eval_mask, label, logger, slice=sample_number, slice_num=split_version, save_pt_name=run_name)
                    save_cluster(sample_number, test_ari, test_ari_ref, test_nmi, test_nmi_ref, num_layers, max_epoch, output_dir=args.output_dir)
                
                    test_aris.append(test_ari)
                    test_nmis.append(test_nmi)
                
                if args.task == "imputation":
                    test_l1, test_cos_sim, test_rmse = model.batch_test(args, graphs, eval_mask, label, logger, slice=sample_number, slice_num=split_version, save_pt_name=run_name)
                    save_imputation(sample_number, test_l1, test_cos_sim, test_rmse, num_layers, max_epoch, output_dir=args.output_dir)
                    
                    test_l1s.append(test_l1)
                    test_cos_sims.append(test_cos_sim)
                    test_rmses.append(test_rmse)
        
        if args.task == "cluster":
            print("\n")
            print("Test ARI average: ", np.mean(test_aris))
            print("Test ARI std: ", np.std(test_aris))
            print("Test NMI average: ", np.mean(test_nmis))
            print("Test NMI std: ", np.std(test_nmis))
        
            save_cluster_avg(sample_number,  np.mean(test_aris), np.std(test_aris), np.mean(test_nmis), np.std(test_nmis), num_layers, max_epoch, output_dir=args.output_dir)
        
        elif args.task == "imputation":
            print("\n")
            print("Test L1 average: ", np.mean(test_l1s))
            print("Test L1 std: ", np.std(test_l1s))
            print("Test Cosine Similarity average: ", np.mean(test_cos_sims))
            print("Test Cosine Similarity std: ", np.std(test_cos_sims))
            print("Test RMSE average: ", np.mean(test_rmses))
            print("Test RMSE std: ", np.std(test_rmses))
            
            save_imputation_avg(sample_number, test_l1s, test_cos_sims, test_rmses, num_layers, max_epoch, output_dir=args.output_dir)
    
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    np.seterr(divide='ignore', invalid='ignore')
    args = build_args()
    print(args)
    main(args)
