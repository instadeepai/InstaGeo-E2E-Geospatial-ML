import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import Auth0ProviderWrapper from './components/Auth0Provider';
import 'leaflet/dist/leaflet.css';

const container = document.getElementById('root');
const root = createRoot(container);
root.render(
  <React.StrictMode>
    <Auth0ProviderWrapper>
      <App />
    </Auth0ProviderWrapper>
  </React.StrictMode>
);
