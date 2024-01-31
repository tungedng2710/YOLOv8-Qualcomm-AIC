/opt/qti-aic/exec/qaic-exec -aic-hw \
 -aic-hw-version=2.0 \
 -compile-only -convert-to-fp16 \
 -aic-num-cores=14 \
 -m=vehicle_yolov8s_640.onnx \
 -onnx-define-symbol=batch,1 \
 -onnx-define-symbol=height,640 \
 -onnx-define-symbol=width,640 \
 -aic-binary-dir=vehicle_640_bs1_c14