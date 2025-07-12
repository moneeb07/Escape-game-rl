from pygame import Rect

bodies = []
triggers = []

def reset_physics():
    global bodies, triggers
    bodies.clear()
    triggers.clear()

def get_bodies_within_circle(circle_x, circle_y, radius):
    """Get all bodies within a circular area"""
    in_range = []
    for body in bodies:
        if hasattr(body, 'entity'):  # Check if body has entity reference
            # Get the closest point on the rectangle to the circle's center
            closest_x = max(body.entity.x + body.hitbox.x, 
                          min(circle_x, body.entity.x + body.hitbox.x + body.hitbox.width))
            closest_y = max(body.entity.y + body.hitbox.y, 
                          min(circle_y, body.entity.y + body.hitbox.y + body.hitbox.height))
            
            # Calculate distance between the closest point and circle center
            distance_x = circle_x - closest_x
            distance_y = circle_y - closest_y
            distance_squared = distance_x * distance_x + distance_y * distance_y
            
            if distance_squared <= radius * radius:
                in_range.append(body)
    
    return in_range

class PhysicalObj:
    def __init__(self, x, y, width, height):
        self.hitbox = Rect(x, y, width, height)

    def is_colliding_with(self, other):
        # Calculate exact positions of hitboxes
        
        x = self.entity.x + self.hitbox.x
        y = self.entity.y + self.hitbox.y
        other_x = other.entity.x + other.hitbox.x
        other_y = other.entity.y + other.hitbox.y
        

       

        # Collision condition
        if (x < other_x + other.hitbox.width and
            x + self.hitbox.width > other_x and
            y < other_y + other.hitbox.height and
            y + self.hitbox.height > other_y):
            print("Collision detected!")  # Debugging message
            return True
        
        
        return False


class Trigger(PhysicalObj):
    def __init__(self, on, x=0, y=0, width=16, height=16):
        super().__init__(x, y, width, height)
        triggers.append(self)
        self.on = on
    
    def breakdown(self):
        global triggers
        triggers.remove(self)


class Body(PhysicalObj):
    def __init__(self, x=0, y=0, width=32, height=32):
        super().__init__(x, y, width, height)
        bodies.append(self)

    
    def breakdown(self):
        global bodies
        bodies.remove(self)
    
    
    def is_position_valid(self):
        from core.area import area
        x = self.entity.x + self.hitbox.x
        y = self.entity.y + self.hitbox.y
        if area.map.is_rect_solid(x, y, self.hitbox.width, self.hitbox.height):
            return False
        for body in bodies:
            if body != self and body.is_colliding_with(self):
                return False
        return True
    
    