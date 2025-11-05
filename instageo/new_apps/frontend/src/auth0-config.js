// Auth0 Configuration

const getRedirectUri = () => {
  if (process.env.REACT_APP_AUTH0_REDIRECT_URI) {
    return process.env.REACT_APP_AUTH0_REDIRECT_URI;
  }
  return window.location.origin;
};

export const auth0Config = {
  domain: process.env.REACT_APP_AUTH0_DOMAIN || 'auth0-domain.auth0.com',
  clientId: process.env.REACT_APP_AUTH0_CLIENT_ID || 'auth0-client-id',
  audience: process.env.REACT_APP_AUTH0_AUDIENCE || 'auth0-api-identifier',
  redirectUri: getRedirectUri(),
  scope: 'openid profile email offline_access',
};

export const isAuth0Configured = () => {
  return (
    process.env.REACT_APP_AUTH0_DOMAIN &&
    process.env.REACT_APP_AUTH0_DOMAIN !== 'auth0-domain.auth0.com' &&
    process.env.REACT_APP_AUTH0_CLIENT_ID &&
    process.env.REACT_APP_AUTH0_CLIENT_ID !== 'auth0-client-id' &&
    process.env.REACT_APP_AUTH0_AUDIENCE &&
    process.env.REACT_APP_AUTH0_AUDIENCE !== 'auth0-api-identifier'
  );
};
