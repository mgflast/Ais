import glfw

from OpenGL.GL import (
    glBlendFunc, glEnable, glClearColor, glClear, glViewport,
    GL_TRUE, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_BLEND,
    GL_MULTISAMPLE, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
)


class Window:
    def __init__(self, width, height, title):
        self.width = width
        self.height = height
        self.title = title
        self.clear_color = (0.0, 0.0, 0.0, 1.0)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        glfw.window_hint(glfw.SAMPLES, 4)
        self.glfw_window = glfw.create_window(self.width, self.height, self.title, None, None)
        self.focused = True
        glfw.make_context_current(self.glfw_window)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)
        glEnable(GL_MULTISAMPLE)
        if not self.glfw_window:
            glfw.terminate()
            raise Exception("Could not initialize GLFW window.")

        # input vars
        self.cursor_pos = [0, 0]
        self.cursor_pos_previous_frame = [0, 0]
        self.cursor_delta = [0, 0]
        self.cursor_hold = None       # if set to [x, y], on_update pins cursor_pos here (in-place drag gestures)
        self.scroll_delta = [0, 0]
        self.mouse_event = MouseButtonEvent(None, None, None)
        self.key_event = KeyEvent(None, None, None)
        self.mouse_press_duration = 0.0

        # Change flags
        self.window_size_changed = False
        self.window_gained_focus = False
        self.window_gained_focus_buffer = False
        self.reset_event_timer = False

        # Aux
        self.glfw_time = 0.0
        self.time = 0.0
        self.delta_time = 0.0
        self.force_alpha_zero = False
        self.dropped_files = list()

        # default callbacks
        self.set_window_callbacks()

    def set_callbacks(self):
        # note: in current version of pyimgui, callbacks are _either_ to imgui or to app. setting callbacks takes input away from imgui.
        glfw.set_key_callback(self.glfw_window, self.key_callback)
        # glfw.set_char_callback ... not necessary atm
        glfw.set_mouse_button_callback(self.glfw_window, self.mouse_button_callback)
        glfw.set_scroll_callback(self.glfw_window, self.scroll_callback)
        glfw.set_drop_callback(self.glfw_window, self.drop_file_callback)

    def set_mouse_callbacks(self):
        glfw.set_mouse_button_callback(self.glfw_window, self.mouse_button_callback)
        glfw.set_scroll_callback(self.glfw_window, self.scroll_callback)

    def set_window_callbacks(self):
        glfw.set_window_size_callback(self.glfw_window, self.size_changed_callback)
        glfw.set_window_focus_callback(self.glfw_window, self.window_focus_callback)

    def set_key_callbacks(self):
        glfw.set_key_callback(self.glfw_window, self.key_callback)

    def on_update(self):
        current_time = glfw.get_time()
        self.delta_time = current_time - self.time
        self.time = current_time
        self.window_gained_focus = self.window_gained_focus_buffer
        self.window_gained_focus_buffer = False
        self.scroll_delta = [0.0, 0.0]
        self.dropped_files = list()
        if self.force_alpha_zero:
            self.clear_color = (*self.clear_color[0:3], 0.0)
        glClearColor(*self.clear_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.cursor_pos_previous_frame = self.cursor_pos
        self.cursor_pos = list(self.cursor_hold) if self.cursor_hold is not None else list(glfw.get_cursor_pos(self.glfw_window))
        self.cursor_delta = [-self.cursor_pos_previous_frame[0] + self.cursor_pos[0], -self.cursor_pos_previous_frame[1] + self.cursor_pos[1]]
        if self.reset_event_timer:
            self.mouse_press_duration = 0
            self.reset_event_timer = False
        if self.get_mouse_button(glfw.MOUSE_BUTTON_LEFT) or self.get_mouse_button(glfw.MOUSE_BUTTON_RIGHT) or self.get_mouse_button(glfw.MOUSE_BUTTON_MIDDLE):
            self.mouse_press_duration += self.delta_time
        else:
            self.reset_event_timer = True
        if self.focused:
            glfw.poll_events()

    def make_current(self):
        glfw.make_context_current(self.glfw_window)

    def bring_to_front(self):
        glfw.focus_window(self.glfw_window)

    def set_full_viewport(self):
        glViewport(0, 0, self.width, self.height)

    def end_frame(self):
        glfw.swap_buffers(self.glfw_window)

    def size_changed_callback(self, window, width, height):
        self.width = max([width, 1])
        self.height = max([height, 1])
        glViewport(0, 0, self.width, self.height)
        self.window_size_changed = True

    def scroll_callback(self, _, dx, dy):
        self.scroll_delta = [dx, dy]

    def mouse_button_callback(self, _, button, action, mods):
        if action == glfw.PRESS:
            self.mouse_press_duration = 0.0
        self.mouse_event = MouseButtonEvent(button, action, mods)

    def key_callback(self, window, button, scancode, action, mods):
        self.key_event = KeyEvent(button, action, mods)

    def window_focus_callback(self, window, focused):
        if focused:
            self.focused = True
            self.window_gained_focus_buffer = True
        else:
            self.focused = False

    def drop_file_callback(self, window, paths):
        self.dropped_files = paths

    def get_mouse_button(self, button):
        return glfw.get_mouse_button(self.glfw_window, button)

    def get_key_event(self, key, action=glfw.PRESS, mods=0, pop_event=True):
        if self.key_event:
            if self.key_event.check(key, action, mods):
                if pop_event:
                    self.key_event = None
                return True
        return False

    def get_key(self, key):
        return glfw.get_key(self.glfw_window, key)

    def get_mouse_event(self, button, action, mods=0, max_duration=None, pop_event=True):
        if self.mouse_event:
            if self.mouse_event.check(button, action, mods):
                if pop_event:
                    self.mouse_event = None
                if max_duration is not None:  # note: event is popped regardless of max_duration
                    return self.mouse_press_duration < max_duration
                return True
        return False

    def pop_any_mouse_event(self):
        self.mouse_event = None

    def set_force_alpha_zero(self, force=True):
        self.force_alpha_zero = force

    def is_minimized(self):
        return glfw.get_window_attrib(self.glfw_window, glfw.ICONIFIED)

    def set_icon(self, image):
        glfw.set_window_icon(self.glfw_window, 1, image)


class KeyEvent:
    def __init__(self, key, action, mods):
        self.key = key
        self.action = action
        self.mods = mods

    def check(self, requested_key, requested_action, requested_mods=0):
        if self.key == requested_key:
            if self.action == requested_action:
                if self.mods == requested_mods:
                    return True
        return False

    def __str__(self):
        return f"key {self.key} action {self.action} with modifiers {self.mods}"


class MouseButtonEvent:
    def __init__(self, button, action, mods):
        self.button = button
        self.action = action
        self.mods = mods

    def check(self, requested_button, requested_action, requested_mods):
        if self.button == requested_button:
            if self.action == requested_action:
                if self.mods == requested_mods:
                    return True
        return False

    def __str__(self):
        return f"button {self.button}, action {self.action}, mods {self.mods}"
