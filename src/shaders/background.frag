#version 450

layout(location = 0) out vec4 outColor;

void main() {
	outColor = vec4( fract(gl_FragCoord.x / 100), gl_FragCoord.y / 400, 0.2, 1.0 );
}