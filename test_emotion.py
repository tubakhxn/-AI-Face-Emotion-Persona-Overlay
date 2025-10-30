import sys, traceback
sys.path.insert(0, r'c:\Users\Tuba Khan\Downloads\Persona\ai_face_persona')
try:
    from emotion_model import EmotionModel
    print('imported emotion_model')
    em = EmotionModel(mode='image')
    print('created instance')
    res = em.predict([], (480,640))
    print('RESULT:', res)
except Exception as e:
    traceback.print_exc()
    print('ERROR:', e)
