# stuff for preprocessing 

For preprocessing stuff in pascal voc format (xml) which is generated when you are labeling with LabelImg

## Generate TFRecord from images and XM-annotations

```
python scripts/generate_tfrecord.py -i /[directory with annotations and respective images] \
    -l scripts/label_map.pbtxt -o /[name of the new tfrecord file new_data.record]

# example:

python scripts/generate_tfrecord.py -i /input_dir/ -l scripts/label_map.json -o /my_data.tfrecord

``` 