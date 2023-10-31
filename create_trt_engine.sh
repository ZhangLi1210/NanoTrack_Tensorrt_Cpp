#! /bin/bash

BACK_EXAM_ONNX="onnx/nanotrack_backbone_exampler.onnx"
BACK_TEMP_ONNX="onnx/nanotrack_backbone_template.onnx"
HEAD_ONNX="onnx/nanotrack_head.onnx"

# change this path to your tensorrt trtexec path
TRT_BIN="/home/lizhang/TensorRT-8.5.2.2/bin/trtexec"

BACK_EXAM_TRT="engine/nanotrack_backbone_exam.engine"
BACK_TEMP_TRT="engine/nanotrack_backbone_temp.engine"
HEAD_TRT="engine/nanotrack_head.engine"

# VERBOSE="--verbose"
VERBOSE=""

echo "************** Create BackBone("$BACK_EXAM_ONNX") TRT engine **************" 
$TRT_BIN --onnx=$BACK_EXAM_ONNX --saveEngine=$BACK_EXAM_TRT $VERBOSE
echo "************** Create BackBone TRT engine "$BACK_EXAM_TRT" done! **************"

echo "************** Create BackBone("$BACK_TEMP_ONNX") TRT engine **************" 
$TRT_BIN --onnx=$BACK_TEMP_ONNX --saveEngine=$BACK_TEMP_TRT $VERBOSE
echo "************** Create BackBone TRT engine "$BACK_TEMP_TRT" done! **************"

echo "************** Create Head("$HEAD_ONNX") TRT engine **************" 
$TRT_BIN --onnx=$HEAD_ONNX --saveEngine=$HEAD_TRT $VERBOSE
echo "************** Create BackBone TRT engine "$HEAD_TRT" done! **************" 
