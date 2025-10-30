import sys, traceback
sys.path.insert(0, r'c:\Users\Tuba Khan\Downloads\Persona\ai_face_persona')
try:
    import face_detector
    print('face_detector imported')
    import overlay_utils
    print('overlay_utils imported')
    import emotion_model
    print('emotion_model imported')
    print('Imports OK')
except Exception:
    traceback.print_exc()
    print('Import test failed')
