''' Rendering code based on OpenAI gym's, which can depict any environment
    characterized by cyclic and ``biconstrained'' (closed interval) variables. '''

import numpy as np
import pyglet
from gym.envs.classic_control import rendering

class SystemRenderer(rendering.Viewer):
    def __init__(self, n_free, n_cyclic, n_biconstrained):
        self.n_free = n_free
        self.n_biconstrained = n_biconstrained # reward is a line
        self.n_cyclic = n_cyclic
        self.point_size = 10
        if n_cyclic == 1:
            self.stack = 1
        else:
            self.stack = 2
        self.object_height = 200
        self.circle_width = 200
        self.line_width = 50
        free_across = (self.n_free+self.stack-1) // self.stack
        cyclic_across = (self.n_cyclic+self.stack-1) // self.stack
        biconstrained_across = (self.n_biconstrained+self.stack-1) // self.stack
        if self.stack == 1:
            self.reward_width = 50
            self.reward_height = 200
        else:
            self.reward_width = 100
            self.reward_height = 400
        self.timestep = 0
        rendering.Viewer.__init__(self, self.line_width*free_across+self.circle_width*cyclic_across+self.line_width*biconstrained_across+self.reward_width,
                                  self.object_height*self.stack)
    def render_line(self, position, size, value):
        # value is between -1 and 1
        x, y = position
        w, h = size
        # bottom & right boundaries
        self.draw_line((x, self.height-(y+h)), (x+w, self.height-(y+h)), color=(.2, .2, .2))
        self.draw_line((x+w, self.height-y), (x+w, self.height-(y+h)), color=(.2, .2, .2))
        # important locations on the graph
        top = y+h*.1
        bottom = y+h*.9
        left = x+w*.1
        right = x+w*.9
        center = x+w/2
        vcenter = y+h*.5
        vscale = -h*.4
        # background for the line
        self.draw_line((left, self.height-top), (right, self.height-top), color=(.5, .5, .5))
        self.draw_line((left, self.height-bottom), (right, self.height-bottom), color=(.5, .5, .5))
        self.draw_line((center, self.height-top), (center, self.height-bottom), color=(.5, .5, .5))
        # the point itself
        circ = self.draw_circle(radius=self.point_size, color=(0,0,0))
        circ.add_attr(rendering.Transform(translation=(center,self.height-(vcenter+vscale*value))))
    def render_circle(self, position, size, value):
        # value is between -1 and 1
        x, y = position
        w, h = size
        # bottom & right boundaries
        self.draw_line((x, self.height-(y+h)), (x+w, self.height-(y+h)), color=(.2, .2, .2))
        self.draw_line((x+w, self.height-y), (x+w, self.height-(y+h)), color=(.2, .2, .2))
        # important locations on the graph
        xscale = w*.4
        yscale = h*.4
        xcenter = x+w*.5
        ycenter = y+h*.5
        # background for the circle
        circ = self.draw_circle(radius=1., filled=False, color=(.5, .5, .5))
        circ.add_attr(rendering.Transform(translation=(xcenter,self.height-ycenter), scale=(xscale, yscale)))
        # the point itself
        ypos = -np.sin(value)*xscale
        xpos = np.cos(value)*yscale
        circ = self.draw_circle(radius=self.point_size, color=(0,0,0))
        circ.add_attr(rendering.Transform(translation=(xcenter+xpos,self.height-(ycenter+ypos))))
    def render(self, F, C, B, R, prefix=None):
        x = 0
        for group in range(0, self.n_free, self.stack):
            free_to_display = F[group:group+self.stack]
            y = 0
            for free_variable in free_to_display:
                # render this line
                self.render_line((x, y), (self.line_width, self.object_height), free_variable)
                y += self.object_height
            # render boundary
            x += self.line_width
        for group in range(0, self.n_cyclic, self.stack):
            cyclic_to_display = C[group:group+self.stack]
            y = 0
            for cyclic_variable in cyclic_to_display:
                # render this circle
                self.render_circle((x, y), (self.circle_width, self.object_height), cyclic_variable)
                y += self.object_height
            # render boundary
            x += self.circle_width
        for group in range(0, self.n_biconstrained, self.stack):
            biconstrained_to_display = B[group:group+self.stack]
            y = 0
            for biconstrained_variable in biconstrained_to_display:
                # render this line
                self.render_line((x, y), (self.line_width, self.object_height), biconstrained_variable)
                y += self.object_height
            # render boundary
            x += self.line_width
        # render reward
        y = 0
        display_reward = np.tanh(.05*R)
        self.render_line((x, y), (self.reward_width, self.reward_height), display_reward)
        rendering.Viewer.render(self)
        if prefix:
            if hasattr(prefix, 'numpy'):
                prefix = prefix.numpy().decode()
            buf = pyglet.image.get_buffer_manager().get_color_buffer()
            buf.save(f'{prefix}-{self.timestep:05d}.png')
            self.timestep += 1
