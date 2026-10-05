from app.services import voice


def test_voice_that_cannot_read_the_script_language_is_reported():
    # Cas réel : voix française, script en arabe → Edge TTS ne renvoie rien.
    assert voice.voice_language_mismatch("fr-BE-CharlineNeural-Female", "ar-TN") == ("fr", "ar")
    assert voice.voice_language_mismatch("fr-BE-CharlineNeural-Female", "fr-FR") is None
    assert voice.voice_language_mismatch("ar-TN-ReemNeural-Female", "ar-TN") is None


def test_multilingual_custom_or_unknown_cases_are_not_reported():
    assert voice.voice_language_mismatch("en-US-AvaMultilingualNeural-Female", "ar-TN") is None
    assert voice.voice_language_mismatch("fr-FR-DeniseNeural-Female", "") is None
    assert voice.voice_language_mismatch("fr-FR-DeniseNeural-Female", "auto") is None
    assert voice.voice_language_mismatch("siliconflow:FunAudioLLM/CosyVoice2-0.5B:alex-Male", "fr-FR") is None
