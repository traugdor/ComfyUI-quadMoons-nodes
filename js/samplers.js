import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import "../../scripts/widgets.js";

var initialized = false;
var origProps = {}

const doesInputWithNameExist = (node, name) => node.inputs ? node.inputs.some((input) => input.name === name) : false;
function findWidgetByName(node, name){
        return node.widgets.find(w => w.name === name);
}
const HIDDEN_TAG = "qmHide"

function updateWidgets(node, widget) {
    var widgets = ["upscale_method","ratio"];
    if(node.comfyClass === "quadmoonKSampler" && widget.name === "upscale_latent"){
        var wVal = widget.value;
        switch(wVal){
            case "Yes":
                widgets.forEach((item) => {
                    qmToggleWidget(node, findWidgetByName(node,item), true);
                });
                break;
            case "No":
                widgets.forEach((item) => {
                    qmToggleWidget(node, findWidgetByName(node,item), false);
                });
                break;
        }
    }
}

function qmToggleWidget(node, widget, show = false, suffix = "") {
    if (!widget || doesInputWithNameExist(node, widget.name)) return;

    if (!origProps[widget.name]) {
        origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize };
    }

    widget.type = show ? origProps[widget.name].origType : HIDDEN_TAG + suffix;
    widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];

    widget.linkedWidgets?.forEach(w => harrToggleWidget(node, w, ":" + widget.name, show));

    const newHeight = node.computeSize()[1];
    node.setSize([node.size[0], newHeight]);
}

app.registerExtension({
    name: "qmNodes.qmKSampler",
    nodeCreated(node) {
        if (node.comfyClass === "quadmoonKSampler") {
            for (const w of node.widgets || []) {
                let widgetValue = w.value;
                let originalDescriptor = Object.getOwnPropertyDescriptor(w, 'value');
                updateWidgets(node, w);
                Object.defineProperty(w, 'value', {
                    get() {
                        let valueToReturn = originalDescriptor && originalDescriptor.get
                            ? originalDescriptor.get.call(w)
                            : widgetValue;
                        return valueToReturn;
                    },
                    set(newVal) {
                        if (originalDescriptor && originalDescriptor.set) {
                            originalDescriptor.set.call(w, newVal);
                        } else {
                            widgetValue = newVal;
                        }
                        updateWidgets(node, w);
                    }
                });
            }
        }
        if(findWidgetByName(node, "upscale_latent").value == "No"){
            updateWidgets(node, findWidgetByName(node, "upscale_latent"));
        }
        setTimeout(() => { initialized = true; }, 500);
        const newHeight = node.computeSize()[1];
        node.setSize([node.size[0], newHeight]);
    }
});