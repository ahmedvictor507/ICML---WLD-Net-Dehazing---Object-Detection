To run the gui run:  ./run.sh

To convert the ICEML model:
bash
python convert_to_trt.py --model-type iceml --src models/iceml/final_dehazing_model.pth --width 512 --height 512 --workspace 512

To convert a WLD-Net model (e.g., O-HAZE):
bash
python convert_to_trt.py --model-type wldnet --src models/wldnet/OH-dehazing_model_final.pth --width 512 --height 512 --workspace 512