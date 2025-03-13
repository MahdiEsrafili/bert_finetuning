import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("mtzig/cross_entropy_loss")
launch_gradio_widget(module)