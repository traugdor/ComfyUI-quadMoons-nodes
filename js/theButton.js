import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

var initialized = false;

function killOrRestart(args) {
    var currentNode = app.runningNodeId;
    if(!!currentNode){
        api.interrupt(); //will restart if auto Queue is on. :)
    }
}

function rebootComfy(args) {
    if (confirm("This will REBOOT ComfyUI. Are you sure?")) {
		try {
			api.fetchApi("/manager/reboot");
		}
		catch(exception) {
            alert("Reboot failed. Please check console logs for potential errors.");
		}
	}
}

function startComfy(args) {
        console.log("Queueing prompt")
        app.queuePrompt(-1);
}

app.registerExtension({
    name:"quadmoonsNodes.theButton",
    nodeCreated(node) {
        if(node.comfyClass == "quadmoonThebutton"){
            const restartQueueButton = node.addWidget("button", "KILL or RESTART queue", "", (event) => {
                killOrRestart(event);
            });
            const rebootComfyButton = node.addWidget("button", "REBOOT ComfyUI", "", (event) => {
                rebootComfy(event);
            });
            const startComfyButton = node.addWidget("button", "START queue", "", (event) => {
                startComfy(event);
            });
        }
        for (const w of node.widgets || []) {
            let widgetValue = w.value;

            // Store the original descriptor if it exists
            let originalDescriptor = Object.getOwnPropertyDescriptor(w, 'value');
            Object.defineProperty(w, 'value', {
                get() {
                    // If there's an original getter, use it. Otherwise, return widgetValue.
                    let valueToReturn = originalDescriptor && originalDescriptor.get
                        ? originalDescriptor.get.call(w)
                        : widgetValue;

                    return valueToReturn;
                },
                set(newVal) {
                    // If there's an original setter, use it. Otherwise, set widgetValue.
                    if (originalDescriptor && originalDescriptor.set) {
                        originalDescriptor.set.call(w, newVal);
                    } else {
                        widgetValue = newVal;
                    }
                }
            });
        }
        setTimeout(() => { initialized = true; }, 500);
    },
});