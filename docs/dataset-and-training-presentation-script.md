# Presentation script: the dataset and model training

> Historical ESPDet-Pico study. For the current unchanged YOLO11n evaluation, use [the research report](evaluation_results.md) and [notebook](../notebooks/yolo11n_pretrained_evaluation.ipynb). No YOLO fine-tuning is part of that experiment.

**Suggested duration:** approximately 8–10 minutes, depending on pace and questions.
Read the quoted paragraphs aloud. Bracketed directions are actions for the
presenter. The accompanying [paper section](paper-dataset-and-training.md)
contains formal wording, tables, and references.

## 1. Introduce the two parts of the project

[Show the project overview.]

> Good day. I will explain where our dataset came from, how we prepared it,
> how we trained our original detector, and how we evaluated the results.
>
> First, our project has two distinct model setups. Our original experiment
> fine-tuned a small ESPDet-Pico detector for bottles, cups, and boxes. Our
> current laptop demonstration uses pretrained YOLO11n and a separate depth
> model for broader visual interpretation. The dataset and training results
> I will discuss belong to the original ESPDet experiment. We did not train
> the current YOLO11n or depth model using these 402 images.

## 2. Explain where the images came from

[Open the [public dataset page](https://github.com/thebaynal/Vision-Drive/blob/main/data/public/README.md#preview-and-download).]

> Our images and existing bounding-box annotations came from Open Images.
> The group did not photograph this initial dataset or manually draw its
> original annotations. We selected and prepared a subset of public images.
>
> Specifically, our preparation script used the Open Images validation image
> pool and its annotation metadata. We selected examples of Bottle, Coffee
> cup, and Box, then mapped those source labels to bottle, cup, and box in
> our project. The original image links, authors, license links, and source
> IDs are preserved in our manifest and attribution files.
>
> Open Images supplied the source validation pool. We then created our own
> project training, validation, and testing partitions from that pool. Our
> results are therefore project-subset results, not official Open Images
> benchmark results.

## 3. Explain the size and annotations

[Show the annotated preview and point to one rectangle.]

> The prepared dataset contains 402 images and 688 annotated objects. These
> counts differ because one image may contain several objects. There are
> 179 bottle-containing images, 155 cup-containing images, and 68 box-containing
> images in this prepared subset. Across those images, there are 385 bottle
> boxes, 189 cup boxes, and 114 box boxes.
>
> A bounding box identifies the location of an object in an image. Each image
> has a matching text file in YOLO format. Each line contains a class ID,
> the horizontal and vertical center of a box, its width, and its height.
> Coordinates are normalized between zero and one so that they describe
> proportions of the image rather than a fixed number of pixels.
>
> Our class IDs are zero for bottle, one for cup, and two for box. Using YOLO
> annotation format does not mean the trained architecture must be YOLO11n.
> Our ESPDet training pipeline uses that same dataset format.

## 4. Explain cleaning and dataset splitting

[Show the split table in the dataset card.]

> Before training, our preparation code checked that image files existed,
> that bounding boxes had valid coordinates, and that source information
> was available. It excluded boxes flagged as depictions or groups. It also
> used content hashes to identify duplicate image files before assigning
> project splits.
>
> The final split contains 280 training images, 60 validation images, and
> 62 testing images. That is approximately a 70–15–15 distribution.
>
> The training set is used to update the model's weights. The validation set
> is used to monitor training, choose the checkpoint, and select the detection
> confidence threshold. The testing set is used afterward to evaluate the
> selected model. Test images were not used to choose the original model's
> checkpoint or confidence threshold.
>
> We preserved the source labels while splitting and used a fixed random seed
> for repeatability. The preparation procedure also gave scarce classes a
> place in each partition. Our checks establish distinct file content across
> the splits; they do not prove that every photo shows a different scene or
> a different physical object.

## 5. Explain what “unseen situations” means

> An unseen image is not automatically an unseen situation. Two different
> photos may still show the same room, background, or physical object.
> Therefore, a random image split alone is not enough to demonstrate
> generalization to a new vehicle course.
>
> For a stronger evaluation, we need recordings from the actual camera height,
> with different lighting, distances, backgrounds, angles, and obstacle
> arrangements. Entire recording sessions should stay in one split so that
> nearby frames do not appear in both training and testing.
>
> We also need verified clear-floor examples and unfamiliar obstacles. Our
> current baseline contains target-object images and no dedicated verified
> empty-floor negative set. It contains bounding-box labels, not depth maps
> or correct steering instructions. These are limitations we will report,
> rather than claiming the dataset covers everything in front of the camera.

## 6. Explain the model and transfer learning

[Show the original training notebook or the model diagram.]

> The original model was ESPDet-Pico, using a 224 by 224 input and approximately
> 361 thousand parameters. We selected this architecture for the initial
> embedded-oriented experiment.
>
> We used transfer learning instead of training entirely from random weights.
> The initialization came from Espressif's published single-class checkpoint.
> That source checkpoint was trained for cats. It supplied reusable weights;
> it was not the final bottle, cup, and box detector.
>
> The training code rebuilt the architecture with a three-class detection head
> and transferred compatible weights before fine-tuning. The resulting
> detector's output classes are bottle, cup, and box. It does not keep a cat
> output class. The original checkpoint and our fine-tuned checkpoint are
> separate artifacts with separate recorded hashes.

## 7. Explain how the training ran

[Show the [training curves](../assets/baseline-training-curves.png).]

> Training was configured for a maximum of 60 epochs, with a batch size of
> 16 and random seed 42, running on the CPU. A batch is a group of training
> images processed together; an epoch is one pass through the training set.
>
> The automatic optimizer setting selected AdamW. During training, the model
> adjusted its weights to reduce classification and bounding-box regression
> errors. Classification concerns which object category is present; box
> regression concerns where the object is located.
>
> The training pipeline used resizing and training-time augmentations,
> including horizontal flips, translation, scaling, color variation, and
> mosaic augmentation. These modify training examples and their boxes
> together. The stored original images and the fixed split assignments
> remain unchanged. Augmentation does not create additional independent
> validation or testing images.
>
> After each epoch, the trainer measured validation performance. Early stopping
> had a patience of 15 epochs. The run completed 55 epochs, and the best
> checkpoint was selected at epoch 40. Training stopped because the selection
> score had not improved over the following 15 epochs. We use the best
> checkpoint, rather than assuming the final epoch is best.

## 8. Explain threshold selection and testing

> After selecting the checkpoint, we compared confidence thresholds on the
> validation set. The tested values ranged from 0.15 to 0.50. The evaluation
> script selected 0.35 using macro F1, which averages the F1 score across the
> three classes. It did not select that value from the testing set.
>
> For the reported precision, recall, and F1 table, a prediction was matched
> to a ground-truth box of the same class if their intersection over union,
> or IoU, was at least 0.5. Each target could match only one prediction.
> IoU measures overlap between predicted and annotated boxes.
>
> Precision asks how many predicted objects were correct. Recall asks how
> many annotated objects were found. F1 summarizes the balance between them.
> We also report mean average precision, or mAP, which summarizes detection
> performance across confidence levels. These are different metrics, and
> none is the percentage of successful physical driving trials.

## 9. Present the results honestly

[Show the baseline results table.]

> On the 62 project test images, baseline mAP at IoU 0.5 was approximately
> 0.469, and mAP averaged across IoU thresholds from 0.5 to 0.95 was 0.357.
>
> At confidence 0.35, bottle precision was about 0.559 and recall was 0.442.
> Cup precision and recall were both 0.760. Box precision was 0.222 and
> recall was 0.154. Cups performed best in this run, while box detection
> remained weak.
>
> Low box recall means the detector missed many annotated boxes. It does
> not mean the system can reliably avoid boxes. The smaller box group,
> annotation quality, and differences from a vehicle camera are possible
> contributors, but we have not isolated their effects experimentally.
>
> These numbers belong to our original ESPDet detector. They must not be
> presented as the accuracy of the current YOLO-plus-depth demonstration.

## 10. Connect the experiment to the current demo

> The current demo uses pretrained YOLO11n for common object categories and
> a pretrained depth model for estimated distances and nearby surfaces.
> We have not fine-tuned these two models on our 402-image dataset. Their
> original upstream training data is separate from the dataset we prepared.
>
> Movement instructions are generated by our programmed region and proximity
> rules, not by training on steering labels. The laptop performs perception
> and decision-making. The ESP32 is intended to receive commands for motor
> control. Physical board and driving validation are still outstanding.
>
> Our completed contribution is a prepared dataset, a documented baseline
> fine-tuning experiment, measured detector results, and a working visual
> interpretation and instruction prototype. Our next step is independent
> course data and calibration, followed by physical integration and testing.

## Quick answers for the question-and-answer portion

**Did the group take the photographs?** No. We prepared a subset of public
Open Images data and preserved the original attribution.

**Did the group draw the original boxes?** No. We converted and checked the
provided source annotations; a comprehensive manual label audit remains future work.

**What model did the project fine-tune?** ESPDet-Pico with three output classes.
The current YOLO11n and depth models are pretrained models, not trained on this set.

**Why only 402 images?** This was a bounded initial experiment, not a claim that
402 images are sufficient for general obstacle detection. More representative
data and independently held-out scenarios are needed.

**Does a 0.469 mAP mean 46.9% driving accuracy?** No. It is an object-detection
metric under the reported evaluation protocol, not a driving success rate.

**Can we say it detects every obstacle?** No. Unknown or difficult obstacles may
still be missed, and actual vehicle avoidance has not been validated.

**Can the dataset be downloaded?** Yes: [public dataset page](https://github.com/thebaynal/Vision-Drive/blob/main/data/public/README.md#preview-and-download)
and [dataset release](https://github.com/thebaynal/Vision-Drive/releases/tag/dataset-v1.0.0).
