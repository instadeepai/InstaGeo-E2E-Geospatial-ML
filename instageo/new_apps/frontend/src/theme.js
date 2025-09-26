import { createTheme } from '@mui/material/styles';

export const lightTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#2196f3',
      light: '#64b5f6',
      dark: '#1976d2',
    },
    secondary: {
      main: '#f50057',
    },
    background: {
      default: '#ffffff',
      paper: '#f5f5f5',
    },
    text: {
      primary: '#212121',
      secondary: '#757575',
    },
  },
});

export const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#2196f3',
      light: '#64b5f6',
      dark: '#1976d2',
    },
    secondary: {
      main: '#f50057',
    },
    background: {
      default: '#1a1a2e',
      paper: '#16213e',
    },
    text: {
      primary: '#ffffff',
      secondary: '#e0e0e0',
    },
    divider: '#2196f3',
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          border: '1px solid #2196f3',
          backgroundColor: '#1e2749',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        contained: {
          boxShadow: '0 4px 15px rgba(33, 150, 243, 0.3)',
          '&:hover': {
            boxShadow: '0 6px 20px rgba(33, 150, 243, 0.4)',
          },
        },
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          '&:hover': {
            backgroundColor: '#2196f3',
            color: '#ffffff',
          },
        },
      },
    },
    MuiSlider: {
      styleOverrides: {
        track: {
          backgroundColor: '#2196f3',
        },
        thumb: {
          backgroundColor: '#2196f3',
        },
        rail: {
          backgroundColor: '#e0e0e0',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            backgroundColor: 'transparent',
          },
          '& .MuiInputAdornment-root .MuiSvgIcon-root': {
            color: 'text.primary !important',
          },
          '& input[readonly]': {
            cursor: 'pointer',
            backgroundColor: 'action.hover',
          },
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          backgroundColor: 'transparent',
          color: '#ffffff',
          '& .MuiAlert-icon': {
            color: 'inherit',
          },
        },
        standardSuccess: {
          backgroundColor: 'transparent',
          color: '#4caf50',
          border: '1px solid #4caf50',
        },
        standardError: {
          backgroundColor: 'transparent',
          color: '#f44336',
          border: '1px solid #f44336',
        },
        standardInfo: {
          backgroundColor: 'transparent',
          color: '#2196f3',
          border: '1px solid #2196f3',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          color: '#ffffff',
        },
        colorSuccess: {
          backgroundColor: '#4caf50',
          color: '#ffffff',
        },
        colorError: {
          backgroundColor: '#f44336',
          color: '#ffffff',
        },
        colorPrimary: {
          backgroundColor: '#2196f3',
          color: '#ffffff',
        },
        colorDefault: {
          backgroundColor: '#757575',
          color: '#ffffff',
        },
      },
    },
  },
});
