import React from 'react';
import { Auth0Provider } from '@auth0/auth0-react';
import { auth0Config, isAuth0Configured } from '../auth0-config';
import { logger } from '../utils/logger';

const Auth0ProviderWrapper = ({ children }) => {
  // If Auth0 is not configured, render children without Auth0Provider
  if (!isAuth0Configured()) {
    logger.warn('Auth0 is not configured. Please set up your environment variables.');
    return <>{children}</>;
  }

  return (
    <Auth0Provider
      domain={auth0Config.domain}
      clientId={auth0Config.clientId}
      authorizationParams={{
        redirect_uri: auth0Config.redirectUri,
        audience: auth0Config.audience,
        scope: auth0Config.scope,
      }}
      useRefreshTokens={true}
      cacheLocation="localstorage"
    >
      {children}
    </Auth0Provider>
  );
};

export default Auth0ProviderWrapper;
