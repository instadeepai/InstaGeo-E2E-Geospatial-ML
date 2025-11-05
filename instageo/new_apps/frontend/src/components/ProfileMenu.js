import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Avatar,
  IconButton,
  Button,
  Menu,
  MenuItem,
  CircularProgress
} from '@mui/material';
import PersonIcon from '@mui/icons-material/Person';
import LoginIcon from '@mui/icons-material/Login';
import LogoutIcon from '@mui/icons-material/Logout';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import { useAuth0 } from '@auth0/auth0-react';
import { isAuth0Configured, auth0Config } from '../auth0-config';
import { logger } from '../utils/logger';
import { useTheme } from '@mui/material/styles';

const ProfileMenu = ({ appTheme = 'light' }) => {
  const {
    user,
    isAuthenticated,
    isLoading,
    loginWithRedirect,
    logout,
    getAccessTokenSilently
  } = useAuth0();

  const auth0Enabled = isAuth0Configured();
  const theme = useTheme();
  const [profileMenuAnchor, setProfileMenuAnchor] = useState(null);

  const [tokenState, setTokenState] = useState({
    valid: false,
    checking: false
  });

  // Validate token when user state changes
  useEffect(() => {
    if (!auth0Enabled || !isAuthenticated || isLoading) {
      setTokenState({ valid: false, checking: false });
      logger.warn('Token validation failed: Not authenticated');
      return;
    }

    let isComponentMounted = true;

    const validateToken = async () => {
      setTokenState(prev => ({ ...prev, checking: true }));

      try {
        await getAccessTokenSilently({
          authorizationParams: {
            audience: auth0Config.audience,
          },
          detailedResponse: false
        });
        if (isComponentMounted) {
          logger.warn('Token validation successful');
          setTokenState({ valid: true, checking: false });
        }
      } catch (error) {
        if (isComponentMounted) {
          logger.warn('Token validation failed:', error);
          setTokenState({ valid: false, checking: false });
        }
      }
    };

    validateToken();
    return () => {
      isComponentMounted = false;
    };
  }, [isAuthenticated, isLoading, auth0Enabled, getAccessTokenSilently]);

  // Theme-aware styling using MUI theme colors
  const isDark = theme.palette.mode === 'dark';
  const primaryColor = theme.palette.primary.main;
  const primaryDark = theme.palette.primary.dark;

  const themeStyles = {
    paper: {
      border: isDark ? '1px solid #37474f' : '1px solid #e3f2fd',
      backgroundColor: isDark ? '#1e1e1e' : '#f9fbff',
      borderLeft: `4px solid ${primaryColor}`,
    },
    primaryColor: primaryColor,
    textPrimary: isDark ? theme.palette.text.primary : primaryColor,
    textSecondary: theme.palette.text.secondary,
    avatarBg: isDark ? '#37474f' : '#e0e0e0',
    avatarIcon: isDark ? theme.palette.primary.light : '#666666',
    buttonBorder: isDark ? '#37474f' : primaryColor,
    buttonHover: isDark ? 'rgba(33, 150, 243, 0.08)' : 'rgba(33, 150, 243, 0.04)',
    buttonHoverBorder: primaryDark,
  };

  const handleSignIn = () => {
    if (auth0Enabled) {
      loginWithRedirect({
        appState: {
          returnTo: window.location.pathname,
        },
      });
    } else {
      logger.warn('Auth0 not configured. Please set up your environment variables.');
    }
  };

  const handleSignOut = () => {
    if (auth0Enabled) {
      logout({
        logoutParams: {
          returnTo: window.location.origin,
        },
      });
    } else {
      logger.warn('Auth0 not configured. Please set up your environment variables.');
    }
    setProfileMenuAnchor(null);
  };

  const handleProfileMenuOpen = (event) => {
    setProfileMenuAnchor(event.currentTarget);
  };

  const handleProfileMenuClose = () => {
    setProfileMenuAnchor(null);
  };

  return (
    <>
      <Paper
        elevation={0}
        sx={{
          mb: 3,
          p: 2,
          ...themeStyles.paper,
          borderRadius: 2
        }}
      >
        {(isLoading || tokenState.checking) && auth0Enabled ? (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <CircularProgress size={20} sx={{ color: themeStyles.primaryColor }} />
            <Typography variant="body2" sx={{ color: themeStyles.textSecondary }}>
              Loading...
            </Typography>
          </Box>
        ) : isAuthenticated && user && auth0Enabled && tokenState.valid ? (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Avatar
              src={user.picture}
              alt={user.name || user.email}
              sx={{
                bgcolor: themeStyles.primaryColor,
                width: 40,
                height: 40,
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
              }}
            >
              {!user.picture && <PersonIcon sx={{ color: 'white' }} />}
            </Avatar>
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Typography
                variant="subtitle1"
                sx={{
                  fontWeight: 600,
                  color: themeStyles.textPrimary,
                  fontSize: '0.95rem',
                  lineHeight: 1.2
                }}
                noWrap
              >
                {user.name || user.nickname || user.email?.split('@')[0] || 'User'}
              </Typography>
              <Typography
                variant="caption"
                sx={{
                  color: themeStyles.textSecondary,
                  fontSize: '0.75rem',
                  display: 'block',
                  lineHeight: 1.2
                }}
                noWrap
              >
                {user.email}
              </Typography>
            </Box>
            <IconButton
              size="small"
              onClick={handleProfileMenuOpen}
              sx={{ color: themeStyles.primaryColor }}
            >
              <MoreVertIcon />
            </IconButton>
          </Box>
        ) : (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Avatar
              sx={{
                bgcolor: themeStyles.avatarBg,
                width: 40,
                height: 40,
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
              }}
            >
              <PersonIcon sx={{ color: themeStyles.avatarIcon }} />
            </Avatar>
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Typography
                variant="subtitle1"
                sx={{
                  fontWeight: 600,
                  color: themeStyles.textSecondary,
                  fontSize: '0.95rem',
                  lineHeight: 1.2
                }}
              >
                Guest User
              </Typography>
              <Typography
                variant="caption"
                sx={{
                  color: themeStyles.textSecondary,
                  fontSize: '0.75rem',
                  display: 'block',
                  lineHeight: 1.2
                }}
              >
                {auth0Enabled
                  ? 'Sign in to access your workspace'
                  : 'Auth0 not configured - see AUTH0_SETUP.md'
                }
              </Typography>
            </Box>
            {auth0Enabled ? (
              <Button
                variant="outlined"
                size="small"
                startIcon={<LoginIcon />}
                onClick={handleSignIn}
                sx={{
                  borderColor: themeStyles.buttonBorder,
                  color: themeStyles.primaryColor,
                  '&:hover': {
                    borderColor: themeStyles.buttonHoverBorder,
                    backgroundColor: themeStyles.buttonHover
                  }
                }}
              >
                Sign In
              </Button>
            ) : (
              <Typography
                variant="caption"
                sx={{
                  color: 'warning.main',
                  fontSize: '0.7rem',
                  textAlign: 'right'
                }}
              >
                Setup Required
              </Typography>
            )}
          </Box>
        )}
      </Paper>

      <Menu
        anchorEl={profileMenuAnchor}
        open={Boolean(profileMenuAnchor)}
        onClose={handleProfileMenuClose}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'right',
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
      >
        <MenuItem onClick={handleSignOut}>
          <LogoutIcon sx={{ mr: 1, fontSize: '1.2rem' }} />
          Sign Out
        </MenuItem>
      </Menu>
    </>
  );
};

export default ProfileMenu;
