from app._ollama import response_text


def test_response_text_none():
    assert response_text(None) == ""


def test_response_text_dict():
    assert response_text({"response": "hello"}) == "hello"


def test_response_text_dict_missing_key():
    assert response_text({"other": "x"}) == ""


def test_response_text_object():
    obj = type("R", (), {"response": "hello"})()
    assert response_text(obj) == "hello"


def test_response_text_object_falsy_response():
    obj = type("R", (), {"response": None})()
    assert response_text(obj) == ""


def test_response_text_plain_string():
    assert response_text("raw") == "raw"
