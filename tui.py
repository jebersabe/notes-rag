from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, Button
from textual.containers import Container, Horizontal

class ChatNotes(App):
    TITLE = "chatnotes"
    SUB_TITLE = "Chat with your notes."
    CSS_PATH = "styles.css"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="input_box"):
            yield Input(placeholder="Ask the spirit of your notes", id="message_input")
            yield Button(label="Send", variant="success", id="send_button")
        yield Footer()

if __name__ == "__main__":
    app = ChatNotes()
    app.run()
