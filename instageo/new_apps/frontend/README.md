# InstaGeo Frontend
[Still in development]
A React-based web application for geospatial analysis and modeling. The application allows users to draw bounding boxes on a map and run various geospatial models on the selected areas.

## Architecture

**React-based SPA** with interactive mapping:
- **Map Interface**: Leaflet-based interactive map with drawing tools
- **Task Management**: Create and monitor geospatial processing tasks
- **Real-time Updates**: WebSocket-like polling for task status updates
- **Responsive Design**: Material-UI components for modern UI/UX

## Quick Start

### Using Docker (Recommended)

```bash
cd instageo/new_apps
docker-compose -f docker-compose.dev.yml up -d
```

### Manual Setup

```bash
cd frontend
npm install
npm start
```

## Features

- Interactive map interface using Leaflet
- Draw and edit rectangular bounding boxes
- Real-time area calculation and validation
- Support for multiple bounding boxes
- Area constraints (1 km² to 100,000 km²)
- Model selection and parameter configuration
- Task status monitoring and progress tracking

## API Integration

### Backend Communication
- **Base URL**: `http://localhost:8000` (configurable)
- **Endpoints**:
  - `POST /api/run-model` - Submit new processing task
  - `GET /api/task/{task_id}` - Get task status
  - `GET /api/health` - Backend health check
  - `GET /api/queues/status` - Queue monitoring

### Task Flow
1. User draws bounding boxes on map
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
│   │   ├── BoundingBoxInfo.js
│   │   ├── ControlPanel.js
│   │   ├── MapComponent.js
│   │   └── TaskResultPopup.js
│   ├── config.js          # Application configuration
│   ├── constants.js       # Shared constants and configurations
│   ├── index.js           # Application entry point
│   └── App.js            # Main application component
├── package.json           # Project dependencies and scripts
├── Dockerfile.dev         # Development Docker configuration
└── .gitignore            # Git ignore rules
```

## Environment Variables

```env
REACT_APP_API_BASE_URL=http://localhost:8000
```

## Dependencies

### Core
- **React**: UI framework
- **Material-UI**: Component library
- **Leaflet**: Interactive maps
- **react-leaflet**: React wrapper for Leaflet

### Development
- **Create React App**: Build tooling
- **ESLint**: Code linting
- **Prettier**: Code formatting

## Usage

1. **Drawing Bounding Boxes**
   - Use the rectangle drawing tool in the top-left corner
   - Draw rectangles on the map to define areas of interest
   - Edit or delete existing rectangles using the edit controls

2. **Area Validation**
   - Total area must be between 1 km² and 100,000 km²
   - Invalid areas will show warnings and be prevented
   - Real-time area calculation and validation

3. **Model Configuration**
   - Click the analytics icon in the top-right corner
   - Select a model and configure its parameters
   - Run the model on the selected area

4. **Task Monitoring**
   - View task status and progress
   - Monitor processing stages
   - Access results when complete
