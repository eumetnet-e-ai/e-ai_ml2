import json
import os
import traceback
import zipfile

import numpy as np
import torch

import zarr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def main(args_checkpoint=None, ID=None, resolution="R03B05"):
    if args_checkpoint is None:
        assert ID is not None
        args_checkpoint = f"training_config/outputcheckpoint/{ID}/inference-last.ckpt"
    else:
        assert ID is None
        ID = "None"

    local_rank = 0
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    world_size = 1
    model_comm_group = None
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # load checkpoint file
    model = torch.load(args_checkpoint, map_location=device, weights_only=False)


    # set the model to evaluation mode, turn off gradient computation
    torch.set_grad_enabled(False)
    model.eval()

    # extract meta-data from checkpoint:
    with zipfile.ZipFile(args_checkpoint, "r") as f:
        json_file = next((s for s in f.namelist() if os.path.basename(s) == "ai-models.json"), None)
        metadata = json.load(f.open(json_file, "r"))

    # ----------------------------------------------------------------------------------------
    # query some meta-data:
    latitudes = model.graph_data["data"].x[:, 0].cpu().numpy()
    longitudes = model.graph_data["data"].x[:, 1].cpu().numpy()

    lat_deg = np.rad2deg(latitudes)
    lon_deg = np.rad2deg(longitudes)

    variables = [
        (
            val["mars"]
        )
        for key, val in metadata["dataset"]["variables_metadata"].items()
    ]

    number_of_grid_points = metadata["dataset"]["shape"][-1]

    indices_from = metadata["data_indices"]["data"]["input"]["full"]
    indices_to = metadata["data_indices"]["model"]["input"]["full"]
    mapping = dict({i: j for i, j in zip(indices_from, indices_to)})

    input_index_to_variable = {mapping[i]: v for i, v in enumerate(variables) if i in mapping}

    number_of_input_features = len(indices_to)
    #forcings_variables = set([variables[i] for i in metadata["data_indices"]["data"]["input"]["forcing"]])
    #prognostic_variables = set([variables[i] for i in metadata["data_indices"]["data"]["input"]["prognostic"]])

    indices_from = metadata["data_indices"]["data"]["output"]["full"]
    indices_to = metadata["data_indices"]["model"]["output"]["full"]

    mapping = dict({i: j for i, j in zip(indices_from, indices_to)})
    output_index_to_variable = {mapping[i]: v for i, v in enumerate(variables) if i in mapping}

    # prepare input tensor
    input_tensor_numpy = np.full(
        shape=(
            1,
            2, # multi step
            number_of_input_features,
            number_of_grid_points,
        ),  # Note that the last two dimensions are swapped compared to the expected Anemoi input (for performance reasons)
        fill_value=np.nan,
        dtype=np.float32,
    )
    input_tensor_numpy = input_tensor_numpy.swapaxes(-1, -2)  # this is a view in Anemoi order


    precision = torch.float16 if metadata["config"]["training"]["precision"] == "16-mixed" else None

    ds = zarr.open(f"./dwd-dream-eu-archive-{resolution}-20240502-20241017-3h-v3-CBCF.zarr")
    for i_in_time in [-10, -5, 4, 3, 2]:
        date = str(ds.dates[i_in_time])

        for i, var in input_index_to_variable.items():
            data_i = ds.attrs["variables"].index(var["param"])
            input_tensor_numpy[0, 0, :, i] = ds.data[i_in_time, data_i, 0, :]
            input_tensor_numpy[0, 1, :, i] = ds.data[i_in_time+1, data_i, 0, :]
            #plot(ds.data[0, data_i, 0, :], lon_deg, lat_deg, outname=f"{ID}_{date}_in_{var['param']}", )

        input_tensor_torch = torch.from_numpy(input_tensor_numpy).to(device)
        y_pred = predict_step(
            model,
            input_tensor_torch,
            model_comm_group,
            device,
            precision,
            world_size,
        )
        #n_out = y_pred.shape[-1] # use this for all fields
        n_out = 1
        fig = plt.figure(figsize=[15, 1 + 4*n_out])
        fig.suptitle(date)
        for i_out in range(n_out):
            data = y_pred[..., i_out].cpu().numpy().flatten()
            #param = output_index_to_variable[i_out]['param'] # use this for all fields
            param = "CBCF"
            
            vmax = 400 if "top" in param else 1
    
            ax = fig.add_subplot(n_out, 2, 1 + i_out*2, projection=ccrs.PlateCarree())
            ax.set_title(f"y_pred {param}")
            data[data<0] = np.nan
            plot(data, lon_deg, lat_deg, ax=ax, vmin=0, vmax=vmax)
            i_CLST = ds.attrs["variables"].index(param)
    
            ax = fig.add_subplot(n_out, 2, 2 + i_out*2, projection=ccrs.PlateCarree())
            ax.set_title("target")
            data = ds.data[i_in_time, i_CLST, 0, :]
            plot(data, lon_deg, lat_deg, ax=ax, vmin=0, vmax=vmax)
    
        fig.tight_layout()
        fig.savefig(f"{date}_{ID}_y", dpi=200)
        plt.close(fig)

def plot(data, lon_deg, lat_deg, outname=None, ax=None, **kwargs):
    extent = [lon_deg.min(), lon_deg.max(), lat_deg.min(), lat_deg.max()]

    fig = None
    if ax is None:
        fig = plt.figure(figsize=[10, 7])
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    cmap = ax.scatter(lon_deg, lat_deg, c=data, s=5, **kwargs)
    if outname is not None and fig:
        fig.colorbar(cmap)

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    ax.gridlines(draw_labels=True)
    if outname is not None:
        assert fig
        fig.savefig(outname, dpi=200)
    if fig:
        plt.close(fig)

def predict_step(model, input_tensor_torch, model_comm_group, device, precision, world_size):
    with torch.autocast(device_type=device, dtype=precision):
        if world_size == 1:
            return model.predict_step(input_tensor_torch)

        try:
            return model.predict_step(input_tensor_torch, model_comm_group)
        except Exception as e:
            print(f"ERROR in parallel execution: model.predict_step() crashed with {e}.")
            print(traceback.format_exc())
            raise


if __name__ == "__main__":
    import glob

    # Inference with latest checkpoint
    files = glob.glob("training_config/output/checkpoint/*/inference-last.ckpt")
    files_sorted = sorted(files, key=os.path.getctime)

    main(files_sorted[-1])
