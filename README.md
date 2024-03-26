# Assignment for ML Engineer Position at Avataar 
Name: Amartya Bhattacharya

Current Position: Research Fellow at Indian Institute of Science(IISc), Bangalore

## Generation of Coherent Panorama Image

The task involves using a text prompt to generate a coherent 360 degree panorama image using text to image diffusion model. 

### Part 1: Generation of 360 degree Panorama Image

Here a text to image, diffusion-based panorama generation model has been used. The code to get the output is given below
```python
python3 text_image_wo_prior.py
```
**Sample Output**

**Prompt**: "hdri view, a nice basement in a house in New York, in the style of <s0>TOK<s1>"


**Output Image:** 

![output_2](https://github.com/amartyacodes/AvataarAssignment/assets/44440114/4b0a7830-0a53-42e5-9973-a564564aab8a)

But this doesn't lead to a coherent 360 degree panorama. When a renderer is used the image becomes inconsistent. For example the above image can be used in the renderer 
https://renderstuff.com/tools/360-panorama-web-viewer/

### Part 1: Generation of 360 degree Coherent Panorama Image using Depth Prior

Here a text to image, diffusion-based panorama generation model is used with Control Net, that helps to add a prior for generating the image, it may be a Canny Edge prior or a depth prior. This helps in adding spatially localized input conditions to a pretrained text-to-image diffusion model via efficient finetuning. In order to make a coherent 360 degree panorama image a depth prior is added. The depth prior is given below. ![pano_depth](https://github.com/amartyacodes/AvataarAssignment/assets/44440114/01afedf7-27bf-4a96-b294-0896723e1165)
Now using the given code below, we can generate the output that aligns with our interest.
```python
python3 text_image_w_prior.py
```
**Sample Output**

**Prompt**: "hdri view, a nice basement in a house in New York, in the style of <s0>TOK<s1>"


** Output Image:**

![output_2 (1)](https://github.com/amartyacodes/AvataarAssignment/assets/44440114/1400e81b-7363-4d28-b2fb-dd5cfaa79659)


The five outputs by implementing with and without prior text to image diffusion models has been given in their respective folders named ~with_text_prior" and "without_text_prior"
