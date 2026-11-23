"""Shared FreeGLUT lifecycle; no render timers survive window closure."""

from OpenGL import GLUT


def run_window(viewer):
    GLUT.glutInit(())
    GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA | GLUT.GLUT_DOUBLE | GLUT.GLUT_ALPHA | GLUT.GLUT_DEPTH)
    GLUT.glutInitWindowSize(*viewer.window_size)
    GLUT.glutInitWindowPosition(0, 0)
    viewer.window = GLUT.glutCreateWindow(viewer.title)
    GLUT.glutSetOption(GLUT.GLUT_ACTION_ON_WINDOW_CLOSE, GLUT.GLUT_ACTION_GLUTMAINLOOP_RETURNS)
    GLUT.glutIgnoreKeyRepeat(1)
    GLUT.glutDisplayFunc(viewer.draw_GL)
    GLUT.glutIdleFunc(viewer.idle_callback)
    GLUT.glutReshapeFunc(viewer.resize_GL)
    GLUT.glutKeyboardFunc(viewer.key_pressed)
    GLUT.glutMouseFunc(viewer.mouse_func)
    GLUT.glutMotionFunc(viewer.motion_func)
    viewer._init_GL(*viewer.window_size)
    viewer.time_checker.begin()
    GLUT.glutMainLoop()
