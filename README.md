# Spatio-Temporal Data Visualization & Analysis Tool

## 1. Project Background
In the fields of Earth sciences, oceanography, and meteorology, researchers often deal with complex four-dimensional data (Latitude, Longitude, Time, and Variables). Traditional 2D static charts fail to intuitively display the dynamic changes of these data across time and space, especially when analyzing global-scale continuity and local profile features simultaneously. Furthermore, field sampling data is often sparse and irregularly distributed, making direct visualization impossible without preprocessing. This project was born to address these pain points. It aims to provide a lightweight, integrated solution that combines "Sparse Data Interpolation," "3D Interactive Visualization," and "Real-time Profile Analysis." By integrating advanced Radial Basis Function (RBF) interpolation algorithms with high-performance OpenGL rendering engines, it allows researchers to instantly reconstruct continuous global fields from scattered observation data and perform in-depth analysis through interactive slicing, significantly lowering the technical barrier for multidimensional data analysis and improving research efficiency.

## 2. Application Scenarios
This tool is widely applicable to various scenarios involving spatiotemporal data analysis.
- **Oceanographic Research**: Visualizing the dynamic changes in temperature, salinity, and oxygen content of seawater. Researchers can import sparse sensor data from buoys or survey ships, reconstruct the global ocean environment field, and analyze the vertical structure of thermoclines or ocean currents through slicing.
- **Geological Exploration**: In geological surveys, sampling points are often limited. This tool can interpolate stratum information, helping geologists visualize the subsurface structure and geochemical element distribution in 3D space.
- **Meteorological Analysis**: Displaying the evolution of atmospheric pressure, humidity, and wind fields. The time-axis playback function helps meteorologists observe the movement trajectories of weather systems.
- **Environmental Monitoring**: Tracking the diffusion path of pollutants. By comparing normalized data, the correlation between different pollution indicators can be clearly observed.
- **Educational Demonstration**: In geography or physics classrooms, the "Earth Projection Mode" provides students with an intuitive global perspective, helping them understand abstract spatiotemporal concepts.

## 3. Compatible Hardware
The application is optimized for performance and can run smoothly on a wide range of hardware configurations.
- **Processor (CPU)**: A modern multi-core processor (Intel Core i5/i7 or AMD Ryzen 5/7 series) is recommended to accelerate the calculation process of the RBF interpolation algorithm, especially when processing large amounts of sparse data.
- **Graphics Card (GPU)**: Since the visualization core relies on OpenGL, a discrete graphics card (NVIDIA GeForce GTX 1050 or AMD Radeon RX 560 and above) is strongly recommended to ensure smooth frame rates for 3D rendering and interaction. Integrated graphics (such as Intel UHD/Iris Xe) can also run the application but may experience lag when rendering high-resolution grids or complex spherical meshes.
- **Memory (RAM)**: At least 8GB of RAM is recommended. For high-resolution grid interpolation (e.g., grids exceeding 100x200x50), 16GB or more RAM is required to store the intermediate matrix data and the generated dense data structures.
- **Display**: A monitor with 1080p resolution or higher is recommended to accommodate the layout of the 3D view, control panel, and profile analysis chart simultaneously.

## 4. Operating System
This project is built on the cross-platform Python ecosystem and the Qt framework, ensuring broad compatibility.
- **Windows**: Fully supported on Windows 10 and Windows 11 (64-bit). This is the primary development and testing environment, offering the best stability and performance.
- **macOS**: Compatible with macOS Catalina (10.15) and newer versions. Both Intel chips and Apple Silicon (M1/M2/M3) are supported (via Rosetta 2 or native ARM Python environments).
- **Linux**: Supports mainstream Linux distributions such as Ubuntu 20.04+, Fedora, and CentOS. Users need to ensure that the correct graphics drivers and OpenGL libraries are installed. The X11 window system is recommended, while Wayland users may need additional configuration variables.
- **Note**: Regardless of the operating system, a Python 3.9 or higher environment is required to ensure compatibility with the dependencies listed in `requirements.txt`.

## 5. Dependency Environment
The project relies on a set of powerful open-source Python libraries to implement its core functions.
- **Python 3.9+**: The basic runtime environment.
- **PySide6 (Qt for Python)**: Provides the modern graphical user interface (GUI), including windows, layouts, buttons, and event handling. It is the official Python binding for the Qt framework.
- **pyqtgraph**: A high-performance scientific graphics library based on Qt and OpenGL, used here for 3D terrain rendering, point cloud display, and 2D chart plotting.
- **NumPy**: The fundamental package for scientific computing, used for efficient multi-dimensional array operations, coordinate transformations, and numerical calculations.
- **Pandas**: A powerful data analysis library used for reading CSV files, data cleaning, timestamp parsing, and structured data management.
- **SciPy**: Specifically, the `scipy.interpolate` module, which provides the Radial Basis Function (RBF) interpolation algorithm (`RBFInterpolator`), the core engine for converting sparse scattered data into regular grids.

## 6. Installation Process
Follow these steps to deploy the application in your local environment.

1.  **Clone the Repository**:
    First, download the project code to your local machine.
    ```bash
    git clone https://github.com/GeoPyTool/SpatioTemporalViz.git
    cd SpatioTemporalViz
    ```

2.  **Create a Virtual Environment (Recommended)**:
    To avoid conflicts with system libraries, it is recommended to create an isolated Python environment.
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    Use pip to install all required libraries listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If the installation speed is slow, you can use a mirror source (e.g., `-i https://pypi.tuna.tsinghua.edu.cn/simple`).*

4.  **Verify Installation**:
    Ensure all libraries are installed correctly without errors. You can verify by importing them in a Python shell.

## 7. Running & Screenshots
### How to Run
After successful installation, execute the following command in the terminal to start the application:
```bash
python app.py
```
The main window should appear immediately. You can start by clicking the "Load Data (CSV)" button. If you don't have your own data, use the `gen.py` script provided in the project to generate a sample `raw_data.csv` file, then load it. The app will automatically detect the sparse format and ask if you want to interpolate it.

### Screenshots
1.  **Initial Interface**:
    ![Start Interface](images/0-启动界面.png)
2.  **Data Loading & Interpolation**:
    ![Data Loading](images/1-数据加载.png)
3.  **Variable Visualization**:
    ![Visualization](images/2-数据可视化.png)
4.  **Data Normalization**:
    ![Normalization](images/3-数据归一化.png)
5.  **Earth Projection Mode**:
    ![Earth Projection](images/4-地球投影.png)

## 8. License
This project is open-source software licensed under the **GNU General Public License v3.0 (GPLv3)**.

**Permissions**:
- You may copy, distribute and modify the software as long as you track changes/dates in source files.
- Any modifications to or software including (via compiler) GPL-licensed code must also be made available under the GPL along with build & install instructions.

**Conditions**:
- **Source Code Availability**: You must provide the complete source code of your modified version.
- **Same License**: Derived works must be released under the same GPLv3 license.
- **No Warranty**: The software is provided "as is" without warranty of any kind.

This ensures that the project remains free and open for the community, encouraging collaboration and improvement in the field of scientific visualization.
