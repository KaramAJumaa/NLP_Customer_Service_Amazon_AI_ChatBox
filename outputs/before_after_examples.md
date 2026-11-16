# Before / after preprocessing (real utterances from Amazon MASSIVE en-US)

| Category | Intent | Raw text | clean_text (stop-words removed) | norm_text (all words kept) |
|---|---|---|---|---|
| Contraction expansion | `datetime_query` | what's the time in australia | what be time in australia | what be the time in australia |
| Negation preserved | `general_quirky` | i am not tired i am actually happy | not tire actually happy | i be not tire i be actually happy |
| on/off kept (light on) | `iot_hue_lighton` | turn the lights on | turn light on | turn the light on |
| on/off kept (light off) | `iot_hue_lightoff` | turn the lights off please | turn light off please | turn the light off please |
| up/down kept (volume up) | `audio_volume_up` | turn up the volume | turn up volume | turn up the volume |
| up/down kept (volume down) | `audio_volume_down` | it's really loud can you please turn the music down | be really loud can please turn music down | it be really loud can you please turn the music down |
| Time marker (p. m. -> pm) | `alarm_set` | make an alarm for four p. m. | make alarm four pm | make an alarm for four pm |
| Dotted abbreviation | `datetime_convert` | tell me the time in g. m. t. plus five | tell time in gmt plus five | tell me the time in gmt plus five |
| Verb forms (lemmatization) | `play_music` | start playing music from favorites | start play music from favorite | start play music from favorite |
| Wake word 'olly' removed | `audio_volume_mute` | olly pause for ten seconds | pause ten second | pause for ten second |
| Time marker (am after a number) | `alarm_set` | wake me up at five am this week | wake up five am week | wake me up at five am this week |
