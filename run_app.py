import sys, traceback
sys.path.insert(0, r'c:\Users\Tuba Khan\Downloads\Persona\ai_face_persona')
try:
    import main
    print('Starting main()')
    main.main()
except Exception:
    traceback.print_exc()
    print('ERROR launching main')
