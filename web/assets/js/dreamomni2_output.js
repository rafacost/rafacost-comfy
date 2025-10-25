import {app} from "../../scripts/app.js";
import {ComfyWidgets} from "../../scripts/widgets.js";

app.registerExtension({
    name: "rafacostComfy.DreamOmni2_Output_Node",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "DreamOmni2_Output_Node") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                
                // Create a proper multiline text widget
                const widget = ComfyWidgets["STRING"](
                    this, 
                    "text", 
                    ["STRING", {multiline: true}], 
                    app
                ).widget;
                
                widget.inputEl.readOnly = true;
                widget.inputEl.style.opacity = 0.6;
                
                // Make the node taller for better visibility
                const nodeWidth = this.size[0];
                const nodeHeight = this.size[1];
                this.setSize([nodeWidth, Math.max(nodeHeight * 3, 200)]);
                
                return result;
            };

            // Update the widget when execution completes
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                
                const widget = this.widgets?.find(obj => obj.name === "text");
                if (widget && message.text) {
                    // Handle both array and direct string
                    const textValue = Array.isArray(message.text) ? message.text[0] : message.text;
                    widget.value = textValue;
                    
                    // Force update of the input element
                    if (widget.inputEl) {
                        widget.inputEl.value = textValue;
                    }
                }
            };
        }
    },
});