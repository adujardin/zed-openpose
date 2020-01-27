#ifndef __VIEWER_INCLUDE__
#define __VIEWER_INCLUDE__

#include <iostream>
#include <thread>
#include <vector>
#include <mutex>

#include <sl/Camera.hpp>

#include <GL/glew.h>
#include <GL/freeglut.h> 
#include <GL/gl.h>
#include <GL/glut.h>   /* OpenGL Utility Toolkit header */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class GLObject {
public:
    GLObject(sl::Translation position, bool isStatic);
    ~GLObject();

    void addPoint(float x, float y, float z, float r, float g, float b);
    void pushToGPU();
    void clear();

    void setDrawingType(GLenum type);

    void draw();

    void translate(const sl::Translation& t);
    void setPosition(const sl::Translation& p);

    void setRT(const sl::Transform& mRT);

    void rotate(const sl::Orientation& rot);
    void rotate(const sl::Rotation& m);
    void setRotation(const sl::Orientation& rot);
    void setRotation(const sl::Rotation& m);

    const sl::Translation& getPosition() const;

    sl::Transform getModelMatrix() const;

    std::vector<float> m_vertices_;
    std::vector<float> m_colors_;
    std::vector<unsigned int> m_indices_;

    bool isStatic_;

    GLenum drawingType_;
    GLuint vaoID_;
    /* Vertex buffer IDs:
     - [0]: vertices coordinates;
     - [1]: RGB color values;
     - [2]: indices;*/
    GLuint vboID_[3];

    sl::Translation position_;
    sl::Orientation rotation_;
};

class MeshObject {
    GLuint vaoID_;
    GLuint vboID_[2];
    int current_fc;
    GLenum drawingType_;


public:
    MeshObject();
    ~MeshObject();
    sl::float3 clr;

    void updateMesh(std::vector<sl::float3> &vert, std::vector<sl::uint3> &tri, GLenum type);
    void draw();
};

class PeoplesObject {
    GLuint vaoID_;
    GLuint vboID_[3];
    int current_fc;
    std::mutex mtx;
    std::vector<sl::float3> vert;
    std::vector<sl::float3> clr;
    std::vector<unsigned int> m_indices_;
    bool update;
public:
    PeoplesObject();
    ~PeoplesObject();
    //sl::float3 clr;

    void updateMesh();

    void cpy(PeoplesObject &other) {
        mtx.lock();
        vert = other.vert;
        clr = other.clr;
        m_indices_ = other.m_indices_;
        update = other.update;
        mtx.unlock();
    }

    void clear() {
        mtx.lock();
        vert.clear();
        m_indices_.clear();
        mtx.unlock();
    }


    void setVert(std::vector<sl::float3> &vertices, std::vector<sl::float3> &clr);

    void draw();
};

class LineObject {
    GLuint vaoID_;
    GLuint vboID_[2];
    int current_fc;
    std::mutex mtx;
    std::vector<sl::float3> vert;
    std::vector<unsigned int> m_indices_;
    bool update;
public:
    LineObject();
    ~LineObject();
    sl::float3 clr;

    void updateMesh();

    void clear() {
        mtx.lock();
        vert.clear();
        m_indices_.clear();
        mtx.unlock();
    }


    void setVert(std::vector<sl::float3> &vertices);

    void draw();
};

class GLViewer;

class PointObject {
    friend GLViewer;
    GLuint vaoID_;
    GLuint vboID_[3];
    int current_fc;
    std::mutex mtx;
    std::vector<sl::float3> vert;
    std::vector<sl::float3> clr;
    std::vector<unsigned int> m_indices_;
    bool update;
public:
    PointObject();
    ~PointObject();

    void updateMesh();

    void cpy(PointObject &other) {
        mtx.lock();
        vert = other.vert;
        clr = other.clr;
        m_indices_ = other.m_indices_;
        update = other.update;
        mtx.unlock();
    }

    void setVert(std::vector<sl::float3> &pts, std::vector<sl::float3> &clr);

    void draw();
};

class NormObject {
    GLuint vaoID_;
    GLuint vboID_[2];
    int current_fc;
    GLenum drawingType_;

    sl::float3 pts[2];
    sl::uint2 line;

public:
    NormObject();
    ~NormObject();
    sl::float3 clr;

    void updateNorm(sl::float3 &vert, sl::float3 &normal);
    void draw();
};

class MeshTextureObject {
    GLuint vaoID_;
    GLuint vboID_[3];
    int current_fc;

public:
    MeshTextureObject();
    ~MeshTextureObject();

    void loadData(std::vector<sl::float3> &vert, std::vector<sl::float2> &uv, std::vector<sl::uint3> &tri);

    void draw();

    sl::Mat texture;
};


#ifndef M_PI
#define M_PI 3.141592653
#endif


#define SAFE_DELETE( res ) if( res!=NULL )  { delete res; res = NULL; }

#define MOUSE_R_SENSITIVITY 0.015f
#define MOUSE_UZ_SENSITIVITY 0.75f
#define MOUSE_DZ_SENSITIVITY 1.25f
#define MOUSE_T_SENSITIVITY 0.1f
#define KEY_T_SENSITIVITY 0.1f

class CameraGL {
public:

    CameraGL() {

    }

    enum DIRECTION {
        UP, DOWN, LEFT, RIGHT, FORWARD, BACK
    };

    CameraGL(sl::Translation position, sl::Translation direction, sl::Translation vertical = sl::Translation(0, 1, 0)); // vertical = Eigen::Vector3f(0, 1, 0)
    ~CameraGL();

    void update();
    void setProjection(float horizontalFOV, float verticalFOV, float znear, float zfar);
    const sl::Transform& getViewProjectionMatrix() const;

    float getHorizontalFOV() const;
    float getVerticalFOV() const;

    /*
            Set an offset between the eye of the camera and its position.
            Note: Useful to use the camera as a trackball camera with z>0 and x = 0, y = 0.
            Note: coordinates are in local space.
     */
    void setOffsetFromPosition(const sl::Translation& offset);
    const sl::Translation& getOffsetFromPosition() const;

    void setDirection(const sl::Translation& direction, const sl::Translation &vertical);
    void translate(const sl::Translation& t);
    void setPosition(const sl::Translation& p);
    void rotate(const sl::Orientation& rot);
    void rotate(const sl::Rotation& m);
    void setRotation(const sl::Orientation& rot);
    void setRotation(const sl::Rotation& m);

    const sl::Translation& getPosition() const;
    const sl::Translation& getForward() const;
    const sl::Translation& getRight() const;
    const sl::Translation& getUp() const;
    const sl::Translation& getVertical() const;
    float getZNear() const;
    float getZFar() const;

    static const sl::Translation ORIGINAL_FORWARD;
    static const sl::Translation ORIGINAL_UP;
    static const sl::Translation ORIGINAL_RIGHT;

    sl::Transform projection_;
private:
    void updateVectors();
    void updateView();
    void updateVPMatrix();

    sl::Translation offset_;
    sl::Translation position_;
    sl::Translation forward_;
    sl::Translation up_;
    sl::Translation right_;
    sl::Translation vertical_;

    sl::Orientation rotation_;

    sl::Transform view_;
    sl::Transform vpMatrix_;
    float horizontalFieldOfView_;
    float verticalFieldOfView_;
    float znear_;
    float zfar_;
};

class Shader {
public:

    Shader() {
    }
    Shader(GLchar* vs, GLchar* fs);
    ~Shader();

    GLuint getProgramId();

    static const GLint ATTRIB_VERTICES_POS = 0;
    static const GLint ATTRIB_COLOR_POS = 1;
private:
    bool compile(GLuint &shaderId, GLenum type, GLchar* src);
    GLuint verterxId_;
    GLuint fragmentId_;
    GLuint programId_;
};

/*
 * This class manages the window, input events and Opengl rendering pipeline
 */
class GLViewer {
public:
    GLViewer();
    ~GLViewer();
    void exit();
    bool isEnded();
    bool isInitialized();
    void init(bool useTexture = false);

    void loadTexture();

    void update(PeoplesObject &people);
    void update(PointObject &pc);

private:
    std::mutex mtx_people;
    LineObject camRepere;
    LineObject grill;
    PeoplesObject peopleObj;
    PointObject pointcloudObj;


    /*
  Initialize OpenGL context and variables, and other Viewer's variables
     */
    void initialize();
    /*
      Rendering loop method called each frame by glutDisplayFunc
     */
    void render();
    /*
      Everything that needs to be updated before rendering must be done in this method
     */
    void update();
    /*
      Once everything is updated, every renderable objects must be drawn in this method
     */
    void draw();
    /*
      Clear and refresh inputs' data
     */
    void clearInputs();

    static GLViewer* currentInstance_;

    //! Glut Functions CALLBACKs
    static void drawCallback();
    static void mouseButtonCallback(int button, int state, int x, int y);
    static void mouseMotionCallback(int x, int y);
    static void reshapeCallback(int width, int height);
    static void keyPressedCallback(unsigned char c, int x, int y);
    static void keyReleasedCallback(unsigned char c, int x, int y);
    static void idle();

    bool ended_;

    // color settings
    float cr;
    float cg;
    float cb;

    // window size
    int wnd_w;
    int wnd_h;

    bool useTexture_;

    enum MOUSE_BUTTON {
        LEFT = 0,
        MIDDLE = 1,
        RIGHT = 2,
        WHEEL_UP = 3,
        WHEEL_DOWN = 4
    };

    enum KEY_STATE {
        UP = 'u',
        DOWN = 'd',
        FREE = 'f'
    };

    bool mouseButton_[3];
    int mouseWheelPosition_;
    int mouseCurrentPosition_[2];
    int mouseMotion_[2];
    int previousMouseMotion_[2];
    KEY_STATE keyStates_[256];

    CameraGL camera_;
    Shader shader_, shader_pc, shader_people;
    GLuint shMVPMatrixLoc_;
    GLuint shMVPMatrixLoc_pc;
    GLuint shMVPMatrixLoc_people;
    GLuint shColorLoc_;
    sl::Resolution res;
    CUcontext ctx;
    bool initialized_;
};

#endif /* __VIEWER_INCLUDE__ */
