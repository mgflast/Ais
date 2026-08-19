#vertex
#version 420

layout(location = 0) in vec3 xyz;

uniform mat4 vpMat;

uniform float particleSize;
uniform float pixelSize;
uniform vec3 origin;
uniform vec3 particlePosition;
void main()
{
    gl_Position = vpMat * vec4(xyz * particleSize + (particlePosition) * pixelSize - origin.xyz * pixelSize, 1.0);
}

#fragment
#version 420

out vec4 fragColour;

uniform vec3 particleColour;

void main()
{
    fragColour = vec4(particleColour, 1.0);
}