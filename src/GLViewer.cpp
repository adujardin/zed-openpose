#include "GLViewer.hpp"

GLchar* VERTEX_SHADER =
        "#version 330 core\n"
        "layout(location = 0) in vec3 in_Vertex;\n"
        "uniform mat4 u_mvpMatrix;\n"
        "uniform vec3 u_color;\n"
        "out vec3 b_color;\n"
        "void main() {\n"
        "   b_color = u_color;\n"
        "	gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);\n"
        "}";

GLchar* VERTEX_SHADER_CLR =
        "#version 330 core\n"
        "layout(location = 0) in vec3 in_Vertex;\n"
        "layout(location = 1) in vec3 in_Color;\n"
        "uniform mat4 u_mvpMatrix;\n"
        "out vec3 b_color;\n"
        "void main() {\n"
        "   b_color = in_Color;\n"
        "	gl_Position = u_mvpMatrix * vec4(in_Vertex.xyz, 1);\n"
        "}";

GLchar* FRAGMENT_SHADER =
        "#version 330 core\n"
        "in vec3 b_color;\n"
        "layout(location = 0) out vec4 out_Color;\n"
        "void main() {\n"
        "   out_Color = vec4(b_color, 1);\n"
        "}";

GLchar* FRAGMENT_SHADER_PC =
        "#version 330 core\n"
        "in vec3 b_color;\n"
        "layout(location = 0) out vec4 out_Color;\n"
        "void main() {\n"
        "   out_Color = vec4(b_color, 0.85);\n"
        "}";

GLchar* VERTEX_SHADER_TEXTURE =
        "#version 330 core\n"
        "layout(location = 0) in vec3 in_Vertex;\n"
        "layout(location = 1) in vec2 in_UVs;\n"
        "uniform mat4 u_mvpMatrix;\n"
        "out vec2 UV;\n"
        "void main() {\n"
        "   gl_Position = u_mvpMatrix * vec4(in_Vertex, 1);\n"
        "    UV = in_UVs;\n"
        "}\n";

GLchar* FRAGMENT_SHADER_TEXTURE =
        "#version 330 core\n"
        "in vec2 UV;\n"
        "uniform sampler2D texture_sampler;\n"
        "void main() {\n"
        "    gl_FragColor = vec4(texture(texture_sampler, UV).rgb, 1.0);\n"
        "}\n";

GLchar* POINTCLOUD_VERTEX_SHADER =
        "#version 330 core\n"
        "layout(location = 0) in vec4 in_VertexRGBA;\n"
        "uniform mat4 u_mvpMatrix;\n"
        "out vec3 b_color;\n"
        "vec4 packFloatToVec4i(const float value)\n"
        "{\n"
        "  const vec4 bitSh = vec4(256.0*256.0*256.0, 256.0*256.0, 256.0, 1.0);\n"
        "  const vec4 bitMsk = vec4(0.0, 1.0/256.0, 1.0/256.0, 1.0/256.0);\n"
        "  vec4 res = fract(value * bitSh);\n"
        "  res -= res.xxyz * bitMsk;\n"
        "  return res;\n"
        "}\n"
        "vec4 decomposeFloat(const in float value)\n"
        "{\n"
        "   uint rgbaInt = floatBitsToUint(value);\n"
        "	uint bIntValue = (rgbaInt / 256U / 256U) % 256U;\n"
        "	uint gIntValue = (rgbaInt / 256U) % 256U;\n"
        "	uint rIntValue = (rgbaInt) % 256U; \n"
        "	return vec4(rIntValue / 255.0f, gIntValue / 255.0f, bIntValue / 255.0f, 1.0); \n"
        "}\n"
        "void main() {\n"
        // Decompose the 4th channel of the XYZRGBA buffer to retrieve the color of the point (1float to 4uint)
        "   b_color = decomposeFloat(in_VertexRGBA.a).xyz;\n"
        "	gl_Position = u_mvpMatrix * vec4(in_VertexRGBA.xyz, 1);\n"
        "}";

GLchar* POINTCLOUD_FRAGMENT_SHADER =
        "#version 330 core\n"
        "in vec3 b_color;\n"
        "layout(location = 0) out vec4 out_Color;\n"
        "void main() {\n"
        "   out_Color = vec4(b_color, 1);\n"
        "}";


using namespace sl;

GLObject::GLObject(Translation position, bool isStatic) : isStatic_(isStatic) {
    vaoID_ = 0;
    drawingType_ = GL_TRIANGLES;
    position_ = position;
    rotation_.setIdentity();
}

GLObject::~GLObject() {
    if (vaoID_ != 0) {
        glDeleteBuffers(3, vboID_);
        glDeleteVertexArrays(1, &vaoID_);
    }
}

void GLObject::addPoint(float x, float y, float z, float r, float g, float b) {
    m_vertices_.push_back(x);
    m_vertices_.push_back(y);
    m_vertices_.push_back(z);
    m_colors_.push_back(r);
    m_colors_.push_back(g);
    m_colors_.push_back(b);
    m_indices_.push_back(m_indices_.size());
}

void GLObject::pushToGPU() {
    if (!isStatic_ || vaoID_ == 0) {
        if (vaoID_ == 0) {
            glGenVertexArrays(1, &vaoID_);
            glGenBuffers(3, vboID_);
        }
        glBindVertexArray(vaoID_);
        glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
        glBufferData(GL_ARRAY_BUFFER, m_vertices_.size() * sizeof (float), &m_vertices_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
        glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

        glBindBuffer(GL_ARRAY_BUFFER, vboID_[1]);
        glBufferData(GL_ARRAY_BUFFER, m_colors_.size() * sizeof (float), &m_colors_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
        glVertexAttribPointer(Shader::ATTRIB_COLOR_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_COLOR_POS);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[2]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_indices_.size() * sizeof (unsigned int), &m_indices_[0], isStatic_ ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);

        glBindVertexArray(0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}

void GLObject::clear() {
    m_vertices_.clear();
    m_colors_.clear();
    m_indices_.clear();
}

void GLObject::setDrawingType(GLenum type) {
    drawingType_ = type;
}

void GLObject::draw() {
    glBindVertexArray(vaoID_);
    glDrawElements(drawingType_, (GLsizei) m_indices_.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void GLObject::translate(const Translation& t) {
    position_ = position_ + t;
}

void GLObject::setPosition(const Translation& p) {
    position_ = p;
}

void GLObject::setRT(const Transform& mRT) {
    position_ = mRT.getTranslation();
    rotation_ = mRT.getOrientation();
}

void GLObject::rotate(const Orientation& rot) {
    rotation_ = rot * rotation_;
}

void GLObject::rotate(const Rotation& m) {
    this->rotate(sl::Orientation(m));
}

void GLObject::setRotation(const Orientation& rot) {
    rotation_ = rot;
}

void GLObject::setRotation(const Rotation& m) {
    this->setRotation(sl::Orientation(m));
}

const Translation& GLObject::getPosition() const {
    return position_;
}

Transform GLObject::getModelMatrix() const {
    Transform tmp = Transform::identity();
    tmp.setOrientation(rotation_);
    tmp.setTranslation(position_);
    return tmp;
}

MeshObject::MeshObject() {
    current_fc = 0;
    vaoID_ = 0;
}

MeshObject::~MeshObject() {
    current_fc = 0;
    if (vaoID_)
        glDeleteBuffers(2, vboID_);
}

void MeshObject::updateMesh(std::vector<sl::float3> &vert, std::vector<sl::uint3> &tri, GLenum type) {
    if (vaoID_ == 0) {
        glGenVertexArrays(1, &vaoID_);
        glGenBuffers(2, vboID_);
    }
    drawingType_ = type;
    glBindVertexArray(vaoID_);

    glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
    glBufferData(GL_ARRAY_BUFFER, vert.size() * sizeof (sl::float3), &vert[0], GL_DYNAMIC_DRAW);
    glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[1]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, tri.size() * sizeof (sl::uint3), &tri[0], GL_DYNAMIC_DRAW);

    current_fc = tri.size() * 3;

    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void MeshObject::draw() {
    if ((current_fc > 0) && (vaoID_ > 0)) {
        glBindVertexArray(vaoID_);
        glDrawElements(drawingType_, (GLsizei) current_fc, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
}

PeoplesObject::PeoplesObject() {
    current_fc = 0;
    vaoID_ = 0;
    update = false;
}

PeoplesObject::~PeoplesObject() {
    current_fc = 0;
    if (vaoID_)
        glDeleteBuffers(3, vboID_);
}

void PeoplesObject::setVert(std::vector<sl::float3> &vertices, std::vector<sl::float3> &clr) {
    mtx.lock();
    vert = vertices;
    this->clr = clr;

    m_indices_.clear();
    m_indices_.reserve(vertices.size());
    for (int i = 0; i < vertices.size(); i++)
        m_indices_.push_back(i);

    update = true;
    mtx.unlock();
}

void PeoplesObject::updateMesh() {
    if (update) {
        mtx.lock();

        if (vaoID_ == 0) {
            glGenVertexArrays(1, &vaoID_);
            glGenBuffers(3, vboID_);
        }

        glBindVertexArray(vaoID_);

        glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
        glBufferData(GL_ARRAY_BUFFER, vert.size() * sizeof (sl::float3), &vert[0], GL_DYNAMIC_DRAW);
        glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

        glBindBuffer(GL_ARRAY_BUFFER, vboID_[1]);
        glBufferData(GL_ARRAY_BUFFER, clr.size() * sizeof (sl::float3), &clr[0], GL_DYNAMIC_DRAW);
        glVertexAttribPointer(Shader::ATTRIB_COLOR_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_COLOR_POS);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[2]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_indices_.size() * sizeof (unsigned int), &m_indices_[0], GL_DYNAMIC_DRAW);

        current_fc = (int) m_indices_.size();

        glBindVertexArray(0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        mtx.unlock();
    }
    update = false;
}

void PeoplesObject::draw() {
    if ((current_fc > 0) && (vaoID_ > 0) && mtx.try_lock()) {
        glBindVertexArray(vaoID_);
        glLineWidth(4);
        glDrawElements(GL_LINES, (GLsizei) current_fc, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        mtx.unlock();
    }
}

LineObject::LineObject() {
    current_fc = 0;
    vaoID_ = 0;
    update = false;
}

LineObject::~LineObject() {
    current_fc = 0;
    if (vaoID_)
        glDeleteBuffers(2, vboID_);
}

void LineObject::setVert(std::vector<sl::float3> &vertices) {
    mtx.lock();
    vert = vertices;

    m_indices_.clear();
    m_indices_.reserve(vertices.size());
    for (int i = 0; i < vertices.size(); i++)
        m_indices_.push_back(i);

    update = true;
    mtx.unlock();
}

void LineObject::updateMesh() {
    if (update) {
        mtx.lock();
        if (vaoID_ == 0) {
            glGenVertexArrays(1, &vaoID_);
            glGenBuffers(2, vboID_);
        }

        glBindVertexArray(vaoID_);

        glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
        glBufferData(GL_ARRAY_BUFFER, vert.size() * sizeof (sl::float3), &vert[0], GL_DYNAMIC_DRAW);
        glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[1]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_indices_.size() * sizeof (unsigned int), &m_indices_[0], GL_DYNAMIC_DRAW);

        current_fc = (int) m_indices_.size();

        glBindVertexArray(0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        mtx.unlock();
    }
    update = false;
}

void LineObject::draw() {
    if ((current_fc > 0) && (vaoID_ > 0) && mtx.try_lock()) {
        glBindVertexArray(vaoID_);
        glLineWidth(1);
        glDrawElements(GL_LINES, (GLsizei) current_fc, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        mtx.unlock();
    }
}

//_________________________________________________

PointObject::PointObject() {
    current_fc = 0;
    vaoID_ = 0;
    update = false;
}

PointObject::~PointObject() {
    current_fc = 0;
    if (vaoID_)
        glDeleteBuffers(3, vboID_);
}

void PointObject::setVert(std::vector<sl::float3> &pts, std::vector<sl::float3> &clr_) {
    mtx.lock();

    vert = pts;
    clr = clr_;

    if (m_indices_.size() != vert.size()) {
        m_indices_.resize(vert.size());
        for (int i = 0; i < vert.size(); i++)
            m_indices_[i] = i;
    }

    update = true;
    mtx.unlock();
}

void PointObject::updateMesh() {
    if (update) {
        mtx.lock();

        if (vaoID_ == 0) {
            glGenVertexArrays(1, &vaoID_);
            glGenBuffers(3, vboID_);
        }

        glBindVertexArray(vaoID_);

        glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
        glBufferData(GL_ARRAY_BUFFER, vert.size() * sizeof (sl::float3), &vert[0], GL_DYNAMIC_DRAW);
        glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

        glBindBuffer(GL_ARRAY_BUFFER, vboID_[1]);
        glBufferData(GL_ARRAY_BUFFER, clr.size() * sizeof (sl::float3), &clr[0], GL_DYNAMIC_DRAW);
        glVertexAttribPointer(Shader::ATTRIB_COLOR_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(Shader::ATTRIB_COLOR_POS);


        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[2]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_indices_.size() * sizeof (unsigned int), &m_indices_[0], GL_DYNAMIC_DRAW);

        current_fc = (int) m_indices_.size();
        glBindVertexArray(0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        mtx.unlock();
    }
    update = false;
}

void PointObject::draw() {
    if ((current_fc > 0) && (vaoID_ > 0) && mtx.try_lock()) {
        glBindVertexArray(vaoID_);
        glDrawElements(GL_POINTS, (GLsizei) current_fc, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        mtx.unlock();
    }
}

//_________________________________________________

NormObject::NormObject() {
    current_fc = 0;
    vaoID_ = 0;
}

NormObject::~NormObject() {
    current_fc = 0;
    if (vaoID_)
        glDeleteBuffers(2, vboID_);
}

inline sl::float3 applyRot(sl::float3 &pt, sl::Rotation &rot) {
    sl::float3 tmp;
    tmp[0] = pt[0] * rot.r[0] + pt[1] * rot.r[1] + pt[2] * rot.r[2];
    tmp[1] = pt[0] * rot.r[4] + pt[1] * rot.r[5] + pt[2] * rot.r[6];
    tmp[2] = pt[0] * rot.r[8] + pt[1] * rot.r[9] + pt[2] * rot.r[10];
    return tmp;
}

void NormObject::updateNorm(sl::float3 &vert, sl::float3 &normal) {
    if (vaoID_ == 0) {
        glGenVertexArrays(1, &vaoID_);
        glGenBuffers(2, vboID_);
    }

    glBindVertexArray(vaoID_);

    pts[0] = vert;

    sl::Translation t_(1, 0, 0);
    sl::Rotation rot_x(normal.x, t_);
    t_ = sl::Translation(0, 1, 0);
    sl::Rotation rot_y(normal.y, t_);
    t_ = sl::Translation(0, 0, 1);
    sl::Rotation rot_z(normal.z, t_);

    sl::float3 pt(1, 1, 1);
    pt += vert;
    pt = applyRot(pt, rot_x);
    pt = applyRot(pt, rot_y);
    pt = applyRot(pt, rot_z);

    pts[0] = vert;
    pts[1] = pt;

    line.x = 0;
    line.y = 1;

    glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
    glBufferData(GL_ARRAY_BUFFER, 2 * sizeof (sl::float3), &pts[0], GL_DYNAMIC_DRAW);
    glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[1]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof (sl::uint2), &line[0], GL_DYNAMIC_DRAW);

    current_fc = 2;

    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void NormObject::draw() {
    if ((current_fc > 0) && (vaoID_ > 0)) {
        glBindVertexArray(vaoID_);
        glDrawElements(GL_LINES, (GLsizei) current_fc, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
}

MeshTextureObject::MeshTextureObject() {
    current_fc = 0;
    vaoID_ = 0;
}

MeshTextureObject::~MeshTextureObject() {
    current_fc = 0;
}

void MeshTextureObject::loadData(std::vector<sl::float3> &vert, std::vector<sl::float2> &uv, std::vector<sl::uint3> &tri) {
    if (vaoID_ == 0) {
        glGenVertexArrays(1, &vaoID_);
        glGenBuffers(3, vboID_);
    }

    glBindVertexArray(vaoID_);

    glBindBuffer(GL_ARRAY_BUFFER, vboID_[0]);
    glBufferData(GL_ARRAY_BUFFER, vert.size() * sizeof (sl::float3), &vert[0], GL_DYNAMIC_DRAW);
    glVertexAttribPointer(Shader::ATTRIB_VERTICES_POS, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(Shader::ATTRIB_VERTICES_POS);

    glBindBuffer(GL_ARRAY_BUFFER, vboID_[1]);
    glBufferData(GL_ARRAY_BUFFER, uv.size() * sizeof (sl::float2), &uv[0], GL_DYNAMIC_DRAW);
    glVertexAttribPointer(Shader::ATTRIB_COLOR_POS, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(Shader::ATTRIB_COLOR_POS);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboID_[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, tri.size() * sizeof (sl::uint3), &tri[0], GL_DYNAMIC_DRAW);

    current_fc = tri.size() * 3;

    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void MeshTextureObject::draw() {
    if ((current_fc > 0) && (vaoID_ > 0)) {
        glBindVertexArray(vaoID_);
        glDrawElements(GL_TRIANGLES, (GLsizei) current_fc, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
}


GLViewer* GLViewer::currentInstance_ = nullptr;

void getColor(int num_segments, int i, float &c1, float &c2, float &c3) {
    float r = fabs(1. - (float(i)*2.) / float(num_segments));
    c1 = (0.1 * r);
    c2 = (0.3 * r);
    c3 = (0.8 * r);
}

GLViewer::GLViewer() : initialized_(false) {
    if (currentInstance_ != nullptr) {
        delete currentInstance_;
    }
    currentInstance_ = this;

    wnd_w = 1000;
    wnd_h = 1000;

    cb = 0.847058f;
    cg = 0.596078f;
    cr = 0.203921f;

    mouseButton_[0] = mouseButton_[1] = mouseButton_[2] = false;

    clearInputs();
    previousMouseMotion_[0] = previousMouseMotion_[1] = 0;
    ended_ = true;

}

GLViewer::~GLViewer() {
}

void GLViewer::exit() {
    if (initialized_) {
        ended_ = true;
        glutLeaveMainLoop();
    }
}

bool GLViewer::isEnded() {
    return ended_;
}

void GLViewer::init(bool useTexture) {
    useTexture_ = useTexture;
    res = sl::Resolution(1280, 720);
    //get current CUDA context (created by the ZED) for CUDA - OpenGL interoperability
    cuCtxGetCurrent(&ctx);
    initialize();
    //wait for OpenGL stuff to be initialized
    while (!isInitialized()) sl::sleep_ms(1);
}

void GLViewer::initialize() {
    char *argv[1];
    argv[0] = '\0';
    int argc = 1;
    glutInit(&argc, argv);
    glutInitWindowSize(wnd_w, wnd_h);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutCreateWindow("ZED 3D Viewer");

    GLenum err = glewInit();
    if (GLEW_OK != err)
        std::cout << "ERROR: glewInit failed with error: " << glewGetErrorString(err) << "\n";

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Compile and create the shader
    shader_ = Shader(VERTEX_SHADER, FRAGMENT_SHADER);
    shader_people = Shader(VERTEX_SHADER_CLR, FRAGMENT_SHADER);
    shader_pc = Shader(VERTEX_SHADER_CLR, FRAGMENT_SHADER_PC);

    shMVPMatrixLoc_ = glGetUniformLocation(shader_.getProgramId(), "u_mvpMatrix");
    shMVPMatrixLoc_pc = glGetUniformLocation(shader_pc.getProgramId(), "u_mvpMatrix");
    shMVPMatrixLoc_people = glGetUniformLocation(shader_people.getProgramId(), "u_mvpMatrix");
    shColorLoc_ = glGetUniformLocation(shader_.getProgramId(), "u_color");

    // Create the camera
    camera_ = CameraGL(sl::Translation(0, 0, 0), sl::Translation(0, 0, -1));
    camera_.setOffsetFromPosition(sl::Translation(0, 0, 4));
    sl::Rotation cam_rot;
    sl::float3 euler(180, 0, 0);
    cam_rot.setEulerAngles(euler, 0);
    camera_.setRotation(sl::Rotation(cam_rot));
    
    // Map glut function on this class methods
    glutDisplayFunc(GLViewer::drawCallback);
    glutMouseFunc(GLViewer::mouseButtonCallback);
    glutMotionFunc(GLViewer::mouseMotionCallback);
    glutReshapeFunc(GLViewer::reshapeCallback);
    glutKeyboardFunc(GLViewer::keyPressedCallback);
    glutKeyboardUpFunc(GLViewer::keyReleasedCallback);

    std::vector<sl::float3> grillvec;
    float span = 20.f;
    for (int i = (int) -span; i <= (int) span; i++) {
        grillvec.emplace_back();
        grillvec.back() = sl::float3(i, 0, -span);
        grillvec.emplace_back();
        grillvec.back() = sl::float3(i, 0, span);
        grillvec.emplace_back();
        grillvec.back() = sl::float3(-span, 0, i);
        grillvec.emplace_back();
        grillvec.back() = sl::float3(span, 0, i);
    }
    grill.clr = sl::float3(0.33, 0.33, 0.33);
    grill.setVert(grillvec);
    grill.updateMesh();

    initialized_ = true;
    ended_ = false;
}

void GLViewer::update(PeoplesObject &people) {
    mtx_people.lock();
    peopleObj.cpy(people);
    mtx_people.unlock();
}

void GLViewer::update(PointObject &pc_) {
    mtx_people.lock();
    pointcloudObj.cpy(pc_);
    mtx_people.unlock();
}

void GLViewer::render() {
    if (!ended_) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glClearColor(0.12, 0.12, 0.12, 1.0f);
        //glLineWidth(2);
        glPointSize(1);
        update();
        mtx_people.lock();
        draw();
        mtx_people.unlock();
        glutSwapBuffers();
        sl::sleep_ms(10);
        glutPostRedisplay();
    }
}

bool GLViewer::isInitialized() {
    return initialized_;
}

void GLViewer::update() {
    if (keyStates_['q'] == KEY_STATE::UP || keyStates_['Q'] == KEY_STATE::UP || keyStates_[27] == KEY_STATE::UP) {
        currentInstance_->exit();
        return;
    }

    // Rotation of the camera
    if (mouseButton_[MOUSE_BUTTON::LEFT]) {
        camera_.rotate(sl::Rotation((float) mouseMotion_[1] * MOUSE_R_SENSITIVITY, camera_.getRight()));
        camera_.rotate(sl::Rotation((float) mouseMotion_[0] * MOUSE_R_SENSITIVITY, camera_.getVertical() * -1.f));
    }

    // Translation of the camera on its plane
    if (mouseButton_[MOUSE_BUTTON::RIGHT]) {
        camera_.translate(camera_.getUp() * (float) mouseMotion_[1] * MOUSE_T_SENSITIVITY);
        camera_.translate(camera_.getRight() * (float) mouseMotion_[0] * MOUSE_T_SENSITIVITY);
    }

    // Zoom of the camera
    if (mouseWheelPosition_ != 0) {
        float distance = sl::Translation(camera_.getOffsetFromPosition()).norm();
        if (mouseWheelPosition_ > 0 && distance > camera_.getZNear()) { // zoom
            camera_.setOffsetFromPosition(camera_.getOffsetFromPosition() * MOUSE_UZ_SENSITIVITY);
        } else if (distance < camera_.getZFar()) {// unzoom
            camera_.setOffsetFromPosition(camera_.getOffsetFromPosition() * MOUSE_DZ_SENSITIVITY);
        }
    }

    // Translation of the camera on its axis
    if (keyStates_['u'] == KEY_STATE::DOWN) {
        camera_.translate((camera_.getForward()*-1.f) * KEY_T_SENSITIVITY);
    }
    if (keyStates_['j'] == KEY_STATE::DOWN) {
        camera_.translate(camera_.getForward() * KEY_T_SENSITIVITY);
    }
    if (keyStates_['h'] == KEY_STATE::DOWN) {
        camera_.translate(camera_.getRight() * KEY_T_SENSITIVITY);
    }
    if (keyStates_['k'] == KEY_STATE::DOWN) {
        camera_.translate((camera_.getRight()*-1.f) * KEY_T_SENSITIVITY);
    }

    camera_.update();

    clearInputs();
}

void GLViewer::loadTexture() {
    /*   glEnable(GL_TEXTURE_2D);
       for (int i = 0; i < mesh_tex.size(); i++) {
           mesh_tex[i].texture.indice_gl = i;
           glGenTextures(1, &mesh_tex[i].texture.indice_gl);
           glBindTexture(GL_TEXTURE_2D, mesh_tex[i].texture.indice_gl);
           if (mesh_tex[i].texture.data.getChannels() == 3) {
               glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB
                   , mesh_tex[i].texture.data.getWidth()
                   , mesh_tex[i].texture.data.getHeight()
                   , 0, GL_BGR, GL_UNSIGNED_BYTE
                   , mesh_tex[i].texture.data.getPtr<sl::uchar1>());
           } else {
               glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA
                   , mesh_tex[i].texture.data.getWidth()
                   , mesh_tex[i].texture.data.getHeight()
                   , 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE
                   , mesh_tex[i].texture.data.getPtr<sl::uchar1>());
           }
           glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
           glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
       }*/
}

void GLViewer::draw() {
    const sl::Transform vpMatrix = camera_.getViewProjectionMatrix();

    // Simple 3D shader for simple 3D objects
    glUseProgram(shader_.getProgramId());
    // Axis
    glUniformMatrix4fv(shMVPMatrixLoc_, 1, GL_FALSE, sl::Transform::transpose(vpMatrix).m);

    glUniform3fv(shColorLoc_, 1, camRepere.clr.v);
    camRepere.draw();

    glUniform3fv(shColorLoc_, 1, grill.clr.v);
    grill.draw();

    glUseProgram(0);

    glUseProgram(shader_people.getProgramId());
    glUniformMatrix4fv(shMVPMatrixLoc_people, 1, GL_FALSE, sl::Transform::transpose(vpMatrix).m);
    peopleObj.updateMesh();
    peopleObj.draw();

    glUseProgram(0);

    glUseProgram(shader_pc.getProgramId());
    glUniformMatrix4fv(shMVPMatrixLoc_pc, 1, GL_FALSE, sl::Transform::transpose(vpMatrix).m);
    pointcloudObj.updateMesh();
    pointcloudObj.draw();
    glUseProgram(0);
    
}

void GLViewer::clearInputs() {
    mouseMotion_[0] = mouseMotion_[1] = 0;
    mouseWheelPosition_ = 0;
    for (unsigned int i = 0; i < 256; ++i)
        if (keyStates_[i] != KEY_STATE::DOWN)
            keyStates_[i] = KEY_STATE::FREE;
}

void GLViewer::drawCallback() {
    currentInstance_->render();
}

void GLViewer::mouseButtonCallback(int button, int state, int x, int y) {
    if (button < 5) {
        if (button < 3) {
            currentInstance_->mouseButton_[button] = state == GLUT_DOWN;
        } else {
            currentInstance_->mouseWheelPosition_ += button == MOUSE_BUTTON::WHEEL_UP ? 1 : -1;
        }
        currentInstance_->mouseCurrentPosition_[0] = x;
        currentInstance_->mouseCurrentPosition_[1] = y;
        currentInstance_->previousMouseMotion_[0] = x;
        currentInstance_->previousMouseMotion_[1] = y;
    }
}

void GLViewer::mouseMotionCallback(int x, int y) {
    currentInstance_->mouseMotion_[0] = x - currentInstance_->previousMouseMotion_[0];
    currentInstance_->mouseMotion_[1] = y - currentInstance_->previousMouseMotion_[1];
    currentInstance_->previousMouseMotion_[0] = x;
    currentInstance_->previousMouseMotion_[1] = y;
    glutPostRedisplay();
}

void GLViewer::reshapeCallback(int width, int height) {
    glViewport(0, 0, width, height);
    float hfov = currentInstance_->camera_.getHorizontalFOV();
    currentInstance_->camera_.setProjection(hfov, hfov * (float) height / (float) width, currentInstance_->camera_.getZNear(), currentInstance_->camera_.getZFar());
}

void GLViewer::keyPressedCallback(unsigned char c, int x, int y) {
    currentInstance_->keyStates_[c] = KEY_STATE::DOWN;
    glutPostRedisplay();
}

void GLViewer::keyReleasedCallback(unsigned char c, int x, int y) {
    currentInstance_->keyStates_[c] = KEY_STATE::UP;
}

void GLViewer::idle() {
    glutPostRedisplay();
}

Shader::Shader(GLchar* vs, GLchar* fs) {
    if (!compile(verterxId_, GL_VERTEX_SHADER, vs)) {
        std::cout << "ERROR: while compiling vertex shader" << std::endl;
    }
    if (!compile(fragmentId_, GL_FRAGMENT_SHADER, fs)) {
        std::cout << "ERROR: while compiling fragment shader" << std::endl;
    }

    programId_ = glCreateProgram();

    glAttachShader(programId_, verterxId_);
    glAttachShader(programId_, fragmentId_);

    glBindAttribLocation(programId_, ATTRIB_VERTICES_POS, "in_vertex");
    glBindAttribLocation(programId_, ATTRIB_COLOR_POS, "in_texCoord");

    glLinkProgram(programId_);

    GLint errorlk(0);
    glGetProgramiv(programId_, GL_LINK_STATUS, &errorlk);
    if (errorlk != GL_TRUE) {
        std::cout << "ERROR: while linking Shader :" << std::endl;
        GLint errorSize(0);
        glGetProgramiv(programId_, GL_INFO_LOG_LENGTH, &errorSize);

        char *error = new char[errorSize + 1];
        glGetShaderInfoLog(programId_, errorSize, &errorSize, error);
        error[errorSize] = '\0';
        std::cout << error << std::endl;

        delete[] error;
        glDeleteProgram(programId_);
    }
}

Shader::~Shader() {
    if (verterxId_ != 0)
        glDeleteShader(verterxId_);
    if (fragmentId_ != 0)
        glDeleteShader(fragmentId_);
    if (programId_ != 0)
        glDeleteShader(programId_);
}

GLuint Shader::getProgramId() {
    return programId_;
}

bool Shader::compile(GLuint &shaderId, GLenum type, GLchar* src) {
    shaderId = glCreateShader(type);
    if (shaderId == 0) {
        std::cout << "ERROR: shader type (" << type << ") does not exist" << std::endl;
        return false;
    }
    glShaderSource(shaderId, 1, (const char**) &src, 0);
    glCompileShader(shaderId);

    GLint errorCp(0);
    glGetShaderiv(shaderId, GL_COMPILE_STATUS, &errorCp);
    if (errorCp != GL_TRUE) {
        std::cout << "ERROR: while compiling Shader :" << std::endl;
        GLint errorSize(0);
        glGetShaderiv(shaderId, GL_INFO_LOG_LENGTH, &errorSize);

        char *error = new char[errorSize + 1];
        glGetShaderInfoLog(shaderId, errorSize, &errorSize, error);
        error[errorSize] = '\0';
        std::cout << error << std::endl;

        delete[] error;
        glDeleteShader(shaderId);
        return false;
    }
    return true;
}


const sl::Translation CameraGL::ORIGINAL_FORWARD = sl::Translation(0, 0, 1);
const sl::Translation CameraGL::ORIGINAL_UP = sl::Translation(0, 1, 0);
const sl::Translation CameraGL::ORIGINAL_RIGHT = sl::Translation(1, 0, 0);

CameraGL::CameraGL(Translation position, Translation direction, Translation vertical) {
    this->position_ = position;
    setDirection(direction, vertical);

    offset_ = sl::Translation(0, 0, 0);
    view_.setIdentity();
    updateView();
    setProjection(60, 60, 0.01f, 100.f);
    updateVPMatrix();
}

CameraGL::~CameraGL() {

}

void CameraGL::update() {
    if (sl::Translation::dot(vertical_, up_) < 0)
        vertical_ = vertical_ * -1.f;
    updateView();
    updateVPMatrix();
}

void CameraGL::setProjection(float horizontalFOV, float verticalFOV, float znear, float zfar) {
    horizontalFieldOfView_ = horizontalFOV;
    verticalFieldOfView_ = verticalFOV;
    znear_ = znear;
    zfar_ = zfar;

    float fov_y = verticalFOV * M_PI / 180.f;
    float fov_x = horizontalFOV * M_PI / 180.f;

    projection_.setIdentity();
    projection_(0, 0) = 1.0f / tanf(fov_x * 0.5f);
    projection_(1, 1) = 1.0f / tanf(fov_y * 0.5f);
    projection_(2, 2) = -(zfar + znear) / (zfar - znear);
    projection_(3, 2) = -1;
    projection_(2, 3) = -(2.f * zfar * znear) / (zfar - znear);
    projection_(3, 3) = 0;
}

const sl::Transform& CameraGL::getViewProjectionMatrix() const {
    return vpMatrix_;
}

float CameraGL::getHorizontalFOV() const {
    return horizontalFieldOfView_;
}

float CameraGL::getVerticalFOV() const {
    return verticalFieldOfView_;
}

void CameraGL::setOffsetFromPosition(const sl::Translation& o) {
    offset_ = o;
}

const sl::Translation& CameraGL::getOffsetFromPosition() const {
    return offset_;
}

void CameraGL::setDirection(const sl::Translation& direction, const sl::Translation& vertical) {
    sl::Translation dirNormalized = direction;
    dirNormalized.normalize();
    this->rotation_ = sl::Orientation(ORIGINAL_FORWARD, dirNormalized * -1.f);
    updateVectors();
    this->vertical_ = vertical;
    if (sl::Translation::dot(vertical_, up_) < 0)
        rotate(sl::Rotation(M_PI, ORIGINAL_FORWARD));
}

void CameraGL::translate(const sl::Translation& t) {
    position_ = position_ + t;
}

void CameraGL::setPosition(const sl::Translation& p) {
    position_ = p;
}

void CameraGL::rotate(const sl::Orientation& rot) {
    rotation_ = rot * rotation_;
    updateVectors();
}

void CameraGL::rotate(const sl::Rotation& m) {
    this->rotate(sl::Orientation(m));
}

void CameraGL::setRotation(const sl::Orientation& rot) {
    rotation_ = rot;
    updateVectors();
}

void CameraGL::setRotation(const sl::Rotation& m) {
    this->setRotation(sl::Orientation(m));
}

const sl::Translation& CameraGL::getPosition() const {
    return position_;
}

const sl::Translation& CameraGL::getForward() const {
    return forward_;
}

const sl::Translation& CameraGL::getRight() const {
    return right_;
}

const sl::Translation& CameraGL::getUp() const {
    return up_;
}

const sl::Translation& CameraGL::getVertical() const {
    return vertical_;
}

float CameraGL::getZNear() const {
    return znear_;
}

float CameraGL::getZFar() const {
    return zfar_;
}

void CameraGL::updateVectors() {
    forward_ = ORIGINAL_FORWARD * rotation_;
    up_ = ORIGINAL_UP * rotation_;
    right_ = sl::Translation(ORIGINAL_RIGHT * -1.f) * rotation_;
}

void CameraGL::updateView() {
    sl::Transform transformation(rotation_, (offset_ * rotation_) + position_);
    view_ = sl::Transform::inverse(transformation);
}

void CameraGL::updateVPMatrix() {
    vpMatrix_ = projection_ * view_;
}
