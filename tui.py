import sys
from unittest.mock import MagicMock
from main import RAG
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, Button, Markdown
from textual.containers import Horizontal


sys.modules['multiprocessing.synchronize'] = MagicMock()


class ChatNotes(App):
    TITLE = "chatnotes"
    SUB_TITLE = "Chat with your notes."
    CSS_PATH = "styles.css"

    def __init__(self):
        super().__init__()
        self.rag = RAG()

    def compose(self) -> ComposeResult:
        yield Header()
        yield Markdown("Placeholder...", id="response_output")
        with Horizontal(id="input_box"):
            yield Input(placeholder="Ask the spirit of your notes", id="message_input")
            yield Button(label="Send", variant="success", id="send_button")
        yield Footer()

    @on(Button.Pressed, "#send_button")
    async def send_to_llm(self) -> None:
        input_widget = self.query_one("#message_input")
        md_widget = self.query_one("#response_output")
        if not input_widget.value:
            return
        answer = self.rag(question=input_widget.value).response
        md_widget.update(answer)


if __name__ == "__main__":
    app = ChatNotes()
    app.run()
