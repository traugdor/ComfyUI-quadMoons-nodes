import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

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
        }
    },
});