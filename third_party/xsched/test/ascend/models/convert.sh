cd $2
for file in `ls | grep .onnx`
do
    if test -f $file
    then
        slice=${file%.onnx}
        echo "converting $file to $slice.om"
        atc --model=$slice.onnx --framework=5 --output=$slice --soc_version=$1
    fi
done
