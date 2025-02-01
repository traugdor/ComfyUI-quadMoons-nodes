# ComfyUI - quadmoon's Nodes

## Introduction

This is a repository of all the nodes I wanted to have but couldn't find anywhere. I am quite happy with them and will update this repo as I fix bugs and create new nodes. Enjoy!

## Features

### Existing Nodes
* **The BUTTON** - A one-stop-shop for starting or cancelling your queue or rebooting ComfyUI entirely. Something gone wrong with your setup? Hit ***The BUTTON***. It will take care of everything for you. Use responsibly. ***NEW***: Added ability to start queue as well (for 3rd party apps that don't show the main menu)

![image](https://github.com/traugdor/ComfyUI-quadMoons-nodes/assets/6344355/304fb3ab-d363-4809-8d31-901d4c842bb2)

* **CLIP External Text Encoder** - Your regular `CLIP Text Encoder` node but the text to encode with CLIP defaults to an input instead of a text box. No more right-clicking and converting the text widget to an input! This node works best when used with wildcard selectors and other dynamic output nodes for creating rich and dynamic prompts for your images.

![image](https://github.com/traugdor/ComfyUI-quadMoons-nodes/assets/6344355/e69d37a7-55d1-4d8d-a53a-406ab5ea36a9)

* ***Converters***
    * **X to String Converters** - Need to incorporate a value into a string? Maybe a filename input automation? Need to do some math and output the result into a text display? Convert it to a string and easily output it to whatever nodes you need. It even works with the [ComfyMath](https://github.com/evanspearman/ComfyMath) nodes by [evanspearman](https://github.com/evanspearman) so you can easily use a generic `NUMBER` value as an input!
 
    ![image](https://github.com/traugdor/ComfyUI-quadMoons-nodes/assets/6344355/d2c8be0e-f66b-48bb-bdd8-f8b0fa7ce06c)

    * **Normalize Image Dimensions** - Choose between SD1.5 and SDXL image dimension normalization. This node will quickly calculate an appropriate image size for the platform of your choosing. SD1.5 will clamp the size of a single edge to a maximum or minimum of 512 pixels and SDXL will clamp the dimensions to 1024 pixels on a single edge. Giving attention to minimize will clamp the longest edge to the pixel length. An attention to maximize will clamp the shortest edge to the pixel length. This is done so that input images are the appropriate size for the platform you're working with. Shrinking an image will use an area downscale algorithm. Expanding an image will use a bicubic upscale algorithm. These are used for speed and accuracy of the final image.

    ![image](https://github.com/traugdor/ComfyUI-quadMoons-nodes/assets/6344355/e215ef4e-dc53-42bb-9a44-b88a3b3b03aa)

* **INT Conditional Operation** - Choose between two integer inputs on the fly using conditional logical operators to compare them. Works great for returning the largest or smallest input depending on your needs.

![image](https://github.com/traugdor/ComfyUI-quadMoons-nodes/assets/6344355/fc4ce451-a5f7-4151-b81d-c219b8f6fba0)

* **KSampler - Extra Outputs** - All-in-one KSampler node that includes a latent upscaler and seed output for reuse with other samplers. `KSampler - Extra Outputs` may just be what you need to incorporate into your workflow! 

![image](https://github.com/traugdor/ComfyUI-quadMoons-nodes/assets/6344355/bc0b5c57-70ce-4629-a469-56f581a2069d)

* **Change Background (SEGM SEGS)** - This node is designed to work with Impact Pack from ltdrdata (and even borrows some code to ensure full compatibility with the segs output from the SEGM Detector). What it does is take an input image, and change the background using whatever process you determine as defined by the SEGS input. This 2-pass sampler will change the background in one pass and then resample the entire image as a whole to ensure the new background and the foreground are seamlessly blended.
    <br>
    Inputs:
    * Model - Model used to generate the background as determinted by the SEGS input.
    * Model 2 - Model used to redraw the image in the 2nd pass.
    * Positive - Positive conditioning used to generate the background.
    * Positive 2 - Positive conditioning used to redraw the image as a whole.
    * Negative - Negative conditioning used to generate the badkground.
    * Negative 2 - Negative conditioning used to redraw the image as a whole.
    * Image - The desired input image that will be redrawn using the above inputs.
    * VAE - The desired VAE used to encode and decode the image to latent and vice versa.
    * segs_from_SEGM_Detector - SEGS output from Impact Pack SEGM Detector (SEGS) node. This input will be be understood to define the boundaries of the foreground and background of the input image.

Example (image contains workflow using Impact Pack and this node):

![workflow(4)](https://github.com/traugdor/ComfyUI-quadMoons-nodes/assets/6344355/e06ed8ba-979b-4c2e-b23f-5a01c61c7cf4)

Before and After:

<p align="center">
  <img src="https://github.com/traugdor/ComfyUI-quadMoons-nodes/assets/6344355/04ce4886-655c-447f-ae21-7db91e1a7924" alt="before" width="300" hspace="10"/>
  <img src="https://github.com/traugdor/ComfyUI-quadMoons-nodes/assets/6344355/6665c484-5f7b-4bab-8423-965339158d3b" alt="after" width="300" hspace="10"/>
</p>

* **Smart Negative** - Save and reuse common negative prompts that are used with the model of your choice. 
* **Smart Prompt** - Save and reuse common prompt templates that are used with the model of your choice. (***WIP***)
* **A1111 Alternating Prompts** - A special node that will try to alternate between two prompts that contain a `[alternate|this]` style prompt. Parameters include:
  * All normal KSampler parameters
  * *Image Advance* - How many steps from the current step to generate the image before swapping to the next word in the prompt
  * *Weight* - A percentage value describing how much of the image generation should be done with the Image Advance technique. 
  * *New Seed After Steps*: How many steps should be generated before selecting a new seed.
  
  Experimenting with values can achieve different results, but the default values are best for most images.
* **Batch From Latent** - Create a batch of latents for processing from a single input latent
* **KSampler For Each** - Sample a batch of latents sequentially instead of at once. Saves on GPU VRAM for low VRAM devices
* **Get A1111 Prompt from Image** - Export the more common details saved in an image created by A1111. Output information may vary. This node is not needed with images generated by ComfyUI.

### Planned Nodes


## Installation

Install is currently only supported through ComfyUI Manager by use of the `Install via GIT URL` option. If you wish to install manually, instead, open the terminal program of your choosing and navigate to your ComfyUI installation. Enter the following commands:
1. `cd custom_nodes`
2. `git clone https://github.com/traugdor/ComfyUI-quadMoons-nodes.git`

Any time you change the ComfyUI software or custom nodes you will need to reboot ComfyUI to see the changes.

### Uninstallation
To uninstall **ComfyUI-quadMoons-nodes**, browse to your `custom_nodes` folder and delete the `\ComfyUI-quadMoons-nodes` folder. Reboot ComfyUI.

## Support and Contribution

For support, suggestions, or contributions, please visit the [GitHub repository](https://github.com/traugdor/ComfyUI-quadMoons-nodes) and submit an issue/pull request. I value your feedback greatly!

---

*quadMoon's Nodes* is part of the ComfyUI ecosystem. It is my hope that you are able to incorporate most, if not all, of my nodes into your workflows. Enjoy!

