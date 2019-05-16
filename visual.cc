#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

#include <GL/gl.h>
#include <GL/freeglut.h>
#include <GL/freeglut_ext.h>

#include "types.hh"

using namespace autoreg;

typedef float Real;

enum Projection {
	PROJECTION_X = 0,
	PROJECTION_Y = 1,
	PROJECTION_T = 2,
	PROJECTION_NONE = -1
};

Zeta<Real> func;
Vector<Real,3> delta(1,1,1);

const Real ROT_STEP = 90;

Projection proj = PROJECTION_NONE;
bool paused = true;
int timer = 0;     // таймер для демонстрации в реальном времени
int dragX = 0;
int dragY = 0;
int rotation = 0; /// dimension rotation
int part_count = 4;
Real interval_factor = 0;
int tail = 0;



void
draw_vertex(const size3& c, const size3& offset, float alpha) {
	glColor4f(0.85, 0.85, 0.85, alpha);
	glVertex3f(c[1]+offset[1], c[2]+offset[2], func(c));
}

void
drawSeries(size_t t, Projection p, float alpha) {
	const size3& size = func.shape();
	const size3 offset = -size/2;
	int x1 = size[1];
	int y1 = size[2];
	if (p == PROJECTION_NONE) {
		for (int i=0; i<x1; i++) {
			glBegin(GL_LINE_STRIP);
			for (int j=0; j<y1; j++) {
				draw_vertex(size3(t, i, j), offset, alpha);
			}
			glEnd();
		}
		for (int j=0; j<y1; j++) {
			glBegin(GL_LINE_STRIP);
			for (int i=0; i<x1; i++)
				draw_vertex(size3(t, i, j), offset, alpha);
			glEnd();
		}
	} else {
		int len = size[p];
		size3 d(0, 0, 0);
		glBegin(GL_LINE_STRIP);
		for (; d[p]<len; d[p]++) {
			glVertex3f(d[p]*delta[p], func(d), 0);
		}
		glEnd();
	}
}

void draw_axis(const GLfloat v[3]) {
	glColor3fv(v);
	glBegin(GL_LINES);
	glVertex3f(0, 0, 0);
	glVertex3fv(v);
	glEnd();
}

void resetView() {
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glScalef(delta[0], delta[1], 1.0f);
	//float3 offset = (dom.max() - dom.min())*-0.5;
//	glTranslatef(offset[1], offset[2], -100.0f);
	glTranslatef(0, 0, -100.0f);
	glRotatef(-30.0f, 1, 0, 0);
}

void rotate_dimensions() {
	std::clog << "Rotation is disabled" << std::endl;
/*
	dom.min().rot(rotation);
	dom.max().rot(rotation);
	dom.count().rot(rotation);
	timer = min(timer, dom.count()[0]);
	resetView();
	clog << "Rot = " << rotation << endl;
	clog << "Rotated domain = " << dom << endl;
*/
}


void onDisplay() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(0.25, 0.25, 0.25, 1.0);

	const int tl = std::min(tail, timer);
	for (int t=timer-tl, i=1; t<=timer; ++t, ++i) {
		drawSeries(t, proj, i/(tl+1.0));
		//clog << i/(tail+1.0) << endl;
	}

	std::stringstream str;
	str << "t=" << timer << '/' << func.extent(2)-1;

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glDisable(GL_DEPTH_TEST);

	glRasterPos2i(-1, -1);
	glColor3f(1, 1, 1);
	glutBitmapString(GLUT_BITMAP_HELVETICA_18, (const unsigned char*)str.str().c_str());

	glEnable(GL_DEPTH_TEST);
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	glutSwapBuffers();
}

void onMouseButton(int, int, int x, int y) {
	dragX = x;
	dragY = y;
}

void onMouseDrag(int x, int y) {
	glRotatef(x-dragX, 0, 1, 0);
	glRotatef(y-dragY, 1, 0, 0);
	dragX = x;
	dragY = y;
	glutPostRedisplay();
}

void onResize(int w, int h) {
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//gluPerspective(30.0, (GLfloat)w / (GLfloat)h, 0.1, 100000.0f);
	//glOrtho(-10, 10, -10, 10, 0.1, 1000);
// The following code is a fancy bit of math that is eqivilant to calling:
// // gluPerspective( fieldOfView/2.0f, width/height , 0.1f, 255.0f )
// // We do it this way simply to avoid requiring glu.h
	GLfloat zNear = 0.1f;
	GLfloat zFar = 25500.0f;
	GLfloat aspect = float(w)/float(h);
	GLfloat fH = tan( float(60 / 360.0f * 3.14159f) ) * zNear;
	GLfloat fW = fH * aspect;
	glFrustum( -fW, fW, -fH, fH, zNear, zFar );
	glMatrixMode(GL_MODELVIEW);
//	resetView();
}

void onKeyPressed(unsigned char key, int, int) {
	if (key == 27) exit(0);
	if (key == 32) paused = !paused;

	if (key == 'a') glRotatef(ROT_STEP, 1, 0, 0);
	if (key == 's') glRotatef(ROT_STEP, 0, 1, 0);
	if (key == 'd') glRotatef(ROT_STEP, 0, 0, 1);

	if (key == 'z') glRotatef(-ROT_STEP, 1, 0, 0);
	if (key == 'x') glRotatef(-ROT_STEP, 0, 1, 0);
	if (key == 'c') glRotatef(-ROT_STEP, 0, 0, 1);

	if (key == ']') glScalef(1.5f, 1.5f, 1.5f);
	if (key == '[') glScalef(0.9f, 0.9f, 0.9f);

	if (key == 'h') glTranslatef( 2.0f,  0.0f, 0.0f);
	if (key == 'l') glTranslatef(-2.0f,  0.0f, 0.0f);
	if (key == 'k') glTranslatef( 0.0f,  2.0f, 0.0f);
	if (key == 'j') glTranslatef( 0.0f, -2.0f, 0.0f);

/*
 	if (key == 'r') {
		rotation++;
		rotation %= 3;
		rotate_dimensions();
	}
*/
	glutPostRedisplay();
}

void onSpecialKeyPressed(int key, int, int) {

	int mods = glutGetModifiers();
	float lag = 1.0f;

	if (GLUT_ACTIVE_CTRL == (mods & GLUT_ACTIVE_CTRL)) {
		lag *= 10.0f;
	}

	if (key == GLUT_KEY_F1) proj = PROJECTION_NONE;
	if (key == GLUT_KEY_F2) proj = PROJECTION_X;
	if (key == GLUT_KEY_F3) proj = PROJECTION_Y;
	if (key == GLUT_KEY_F4) proj = PROJECTION_T;
	if (key == GLUT_KEY_F5) resetView();

	if (key == GLUT_KEY_UP)   glScalef(1.5f, 1.5f, 1.5f);
	if (key == GLUT_KEY_DOWN) glScalef(0.9f, 0.9f, 0.9f);

/*
  	// dim rotation
	if (GLUT_ACTIVE_SHIFT == (mods & GLUT_ACTIVE_SHIFT)) {
		if (key == GLUT_KEY_LEFT)  if (rotation > 0) rotation--;
		if (key == GLUT_KEY_RIGHT) rotation++;
		rotation %= 3;
		rotate_dimensions();
	}
*/
	if (proj == PROJECTION_NONE) {
		if (paused) {
			if (key == GLUT_KEY_LEFT)  timer -= lag;
			if (key == GLUT_KEY_RIGHT) timer += lag;
			int sz = func.extent(0);
			if (timer >= sz) timer = sz-1;
		}
	}
	else {
		if (key == GLUT_KEY_LEFT)  glTranslatef( 2.0f, 0.0f, 0.0f);
		if (key == GLUT_KEY_RIGHT) glTranslatef(-2.0f, 0.0f, 0.0f);
	}

	glutPostRedisplay();
}

int get_delta_t() {
	return 1000;
}

void onTimer(int) {
    glutTimerFunc(get_delta_t(), onTimer, 0);
	if (!paused) timer++;
	if (timer >= func.extent(0))
		timer = 0;
	glutPostRedisplay();
}

void initOpenGL(int argc, char** argv) {
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInit(&argc, argv);
	int wnd_w = 800;
	int wnd_h = 600;
	int screen_w = glutGet(GLUT_SCREEN_WIDTH);
	int screen_h = glutGet(GLUT_SCREEN_HEIGHT);
	glutInitWindowSize(wnd_w, wnd_h);
	glutInitWindowPosition((screen_w-wnd_w)/2, (screen_h-wnd_h)/2);
    glutCreateWindow("visual");
    glutReshapeFunc(onResize);
    glutDisplayFunc(onDisplay);
    glutKeyboardFunc(onKeyPressed);
    glutSpecialFunc(onSpecialKeyPressed);
    glutMouseFunc(onMouseButton);
    glutMotionFunc(onMouseDrag);
    glutTimerFunc(0, onTimer, 0);
	//glewInit();
	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glEnable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	onResize(800, 600);
	resetView();
}

void read_valarray(std::istream& in) {
	in >> func;
}

void parse_cmdline(int argc, char** argv) {
	using namespace std;
	stringstream cmdline;
	for (int i=1; i<argc; i++)
		cmdline << argv[i] << ' ';
	string file_name;
	string ar;
	while (!(cmdline >> ar).eof()) {
		if (ar == "-r") cmdline >> tail;
		else if (ar == "-t") cmdline >> timer;
		else {
			file_name = ar;
		}
		cmdline >> ws;
	}
	if (!file_name.empty()) {
		clog << "reading " << ar << endl;
		ifstream in(ar.c_str());
		read_valarray(in);
	} else {
		read_valarray(cin);
	}
	if (timer >= func.extent(0)) {
		timer = func.extent(0)-1;
	}
}

int main(int argc, char** argv) {
	parse_cmdline(argc, argv);
	initOpenGL(argc, argv);
	glutPostRedisplay();
	glutMainLoop();
	return 0;
}
