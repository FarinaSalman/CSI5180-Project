from queue import Queue

ui_queue = Queue()


def push_event(event_type, payload=None):
    ui_queue.put({
        "type": event_type,
        "payload": payload or {},
    })


def set_locked(is_locked: bool):
    push_event("set_locked", {"locked": is_locked})


def set_awake(is_awake: bool):
    push_event("set_awake", {"awake": is_awake})


def set_listening(is_listening: bool):
    push_event("set_listening", {"listening": is_listening})


def add_user_message(text: str):
    push_event("add_user_message", {"text": text})


def add_assistant_message(text: str):
    push_event("add_assistant_message", {"text": text})


def clear_history():
    push_event("clear_history")


def start_timer(duration=None):
    payload = {}
    if duration is not None:
        payload["duration"] = duration
    push_event("start_timer", payload)


def pause_timer():
    push_event("pause_timer")


def stop_timer():
    push_event("stop_timer")


def reset_timer():
    push_event("reset_timer")


def open_book(title, pages, page_index=0):
    ui_queue.put({
        "type": "open_book",
        "payload": {
            "title": title,
            "pages": pages,
            "page_index": page_index,
        },
    })


def close_book():
    push_event("close_book")


def next_page():
    push_event("next_page")


def prev_page():
    push_event("prev_page")


def increase_font_size():
    push_event("increase_font_size")


def decrease_font_size():
    push_event("decrease_font_size")


def increase_brightness():
    push_event("increase_brightness")


def decrease_brightness():
    push_event("decrease_brightness")


def toggle_reader_theme():
    push_event("toggle_reader_theme")

def set_input_text(text: str):
    ui_queue.put({
        "type": "set_input_text",
        "payload": {"text": text or ""}
    })