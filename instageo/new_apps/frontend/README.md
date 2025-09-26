# InstaGeo Frontend
[Still in development]
A React-based web application for geospatial analysis and modeling. The application allows users to draw bounding boxes on a map and run various geospatial models on the selected areas.

## Architecture

**React-based SPA** with interactive mapping:
- **Map Interface**: Leaflet-based interactive map with drawing tools
- **Task Management**: Create and monitor geospatial processing tasks
- **Real-time Updates**: WebSocket-like polling for task status updates
- **Responsive Design**: Material-UI components for modern UI/UX
- **Theme System**: Dynamic theme switching with consistent dark mode across all components


## Features

### Core Functionality
- Interactive map interface using Leaflet with dark/light theme support
- Draw and edit rectangular bounding boxes (single box at a time)
- Real-time area calculation and validation
- Area constraints (configurable via environment variables)
- Model selection and parameter configuration
- Task status monitoring and progress tracking
- PDF report generation with charts and visualizations

### UI/UX Features
- **Dark/Light Theme**: Toggle between themes with consistent styling
- **Task Monitor**: Comprehensive task management with filtering and pagination
- **Tasks Filtering**: Filter tasks by model name, status, and search terms
- **Bounding Box Snapshots**: Visual previews of task areas in the task monitor
- **Responsive Design**: Optimized for desktop and mobile devices

### Advanced Features
- **Model Caching**: 24-hour TTL cache for available models
- **PDF Generation**: Export task results with charts and map visualizations
- **Task Layers**: Visualize and manage multiple task results on the map
- **Progress Tracking**: Real-time status updates with detailed stage information

## API Integration

### Backend Communication
- **Base URL**: `http://localhost:8000` (configurable)
- **Endpoints**:
  - `POST /api/run-model` - Submit new processing task
  - `GET /api/task/{task_id}` - Get task status
  - `GET /api/health` - Backend health check
  - `GET /api/queues/status` - Queue monitoring
  - `GET /api/models` - Get available models
  - `GET /api/visualize/{task_id}` - Get visualization data

### Task Flow
1. User draws a single bounding box on map
2. Frontend validates area constraints
3. Task submitted to backend via API
4. Real-time status updates via polling
5. Results displayed when complete

## Development

### Prerequisites
- Node.js (v14 or higher)
- npm or yarn

### Installation
```bash
cd instageo/new_apps/frontend
npm install
```

### Available Scripts
```bash
npm start          # Start development server (port 3000)
npm run build      # Build for production
npm test           # Run tests
npm run eject      # Eject from Create React App
```

### Project Structure
```
frontend/
├── public/                 # Static files
├── src/                    # Source code
│   ├── components/        # React components
│   │   ├── BoundingBoxInfo.js      # Bounding box information display
│   │   ├── BoundingBoxSnapshot.js  # Map snapshot for task monitor
│   │   ├── ControlPanel.js         # Model selection and parameters
│   │   ├── MapComponent.js         # Main map with drawing tools
│   │   ├── TaskLayers.js           # Task layer management
│   │   ├── TaskLayersControl.js    # Task layer controls
│   │   ├── TaskLayersControlWrapper.js # Task layer wrapper
│   │   ├── TaskResultPopup.js      # Task result display
│   │   ├── TasksMonitor.js         # Task monitoring and filtering
│   │   └── VisualizationDialog.js  # Visualization display
│   ├── theme.js           # Theme configurations (light and dark)
│   ├── utils/             # Utility functions
│   │   ├── logger.js      # Logging utility
│   │   ├── modelsCache.js # Model data caching
│   │   └── pdfReport.js   # PDF generation utility
│   │   └── segmentationColors.js   # Color mapping to classes utility
│   ├── config.js          # Application configuration
│   ├── constants.js       # Shared constants and configurations
│   ├── index.js           # Application entry point
│   └── App.js            # Main application component
├── package.json           # Project dependencies and scripts
├── Dockerfile.dev         # Development Docker configuration
├── Dockerfile.prod        # Production Docker configuration
└── .gitignore            # Git ignore rules
```

## Environment Variables

```env
# Area Validation Limits (in km²)
REACT_APP_MIN_AREA_KM2=50
REACT_APP_MAX_AREA_KM2=500

# Development Configuration
REACT_APP_ENV=dev # or prod - Logging enabled only when set to dev
```

## Dependencies

### Core
- **React**: UI framework
- **Material-UI**: Component library with theming
- **Leaflet**: Interactive maps
- **react-leaflet**: React wrapper for Leaflet
- **leaflet-draw**: Drawing tools for Leaflet
- **@mui/x-date-pickers**: Date picker components
- **dayjs**: Date manipulation library

### PDF Generation
- **jspdf**: PDF generation
- **recharts**: Charts generation
- **recharts-to-png**: Chart to PNG conversion

### Development
- **Create React App**: Build tooling
- **ESLint**: Code linting
- **Prettier**: Code formatting

## Usage

### 1. Drawing Bounding Boxes
- Use the rectangle drawing tool in the top-left corner
- Draw a single rectangle on the map to define the area of interest
- Edit or delete the rectangle using the edit controls
- Area validation ensures the box meets size requirements

### 2. Model Configuration
- Click the analytics icon in the top-right corner to open the control panel
- Select a model from the dropdown (with caching for performance)
- Configure data parameters (temporal tolerance, cloud coverage, date)
- Click "Run Model" to submit the task

### 3. Task Monitoring
- Click the list icon to open the task monitor
- View all tasks with filtering by status, model, and search terms
- See visual snapshots of bounding box areas
- Monitor real-time progress and stage details
- Access results and visualizations when complete

### 4. Theme Management
- Toggle between light and dark themes using the theme button
- All components adapt automatically to the selected theme
- Map tiles are filtered for dark mode compatibility

### 5. PDF Reports
- Generate PDF reports for completed tasks
- Includes charts, visualizations, and map data
- Export functionality available in task results
