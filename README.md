# ComfyUI - quadmoon's Nodes

## Introduction

This is a repository of all the nodes I wanted to have but couldn't find anywhere. I am quite happy with them and will update this repo as I fix bugs and create new nodes. Enjoy!

## Features

### Existing Nodes
* **The BUTTON** - A one-stop-shop for cancelling your queue or rebooting ComfyUI entirely. Something gone wrong with your setup? Hit ***The BUTTON***. It will take care of everything for you. Use responsibly.
* **CLIP External Text Encoder** - Your regular `CLIP Text Encoder` node but the text to encode with CLIP defaults to an input instead of a text box. No more right-clicking and converting the text widget to an input! This node works best when used with wildcard selectors and other dynamic output nodes for creating rich and dynamic prompts for your images.
* **X to String Converters** - Need to incorporate a value into a string? Maybe a filename input automation? Need to do some math and output the result into a text display? Convert it to a string and easily output it to whatever nodes you need. It even works with the [ComfyMath](https://github.com/evanspearman/ComfyMath) nodes by [evanspearman](https://github.com/evanspearman) so you can easily use a generic `NUMBER` value as an input!
* **INT Conditional Operation** - Choose between two integer inputs on the fly using conditional logical operators to compare them. Works great for returning the largest or smallest input depending on your needs.

### Planned Nodes
* **KSampler Seed Output** - Need to use the same seed value between samplers? Want to cleanup your workflow so you don't have to have connections between a seed generator everywhere? KSampler with Extra Outputs may just be what you need to incorporate into your workflow! (***WIP***)

## Installation

Install through ComfyUI Manager (Install via GIT URL)

OR

1. `cd custom_nodes`
2. `git clone https://github.com/traugdor/ComfyUI-quadMoons-nodes.git`
3. Reboot ComfyUI

### Uninstallation
To uninstall **ComfyUI-quadMoons-nodes**, browse to your `custom_nodes` folder and delete the `\ComfyUI-quadMoons-nodes` folder. Reboot ComfyUI.

## Support and Contribution

For support, suggestions, or contributions, please visit the [GitHub repository](https://github.com/traugdor/ComfyUI-quadMoons-nodes), submit an issue/pull request, or contact me on Discord (@quadmoon). I value your feedback greatly!

---

*quadMoon's Nodes* is part of the ComfyUI ecosystem. It is my hope that you are able to incorporate most, if not all, of my nodes into your workflows. Enjoy!

