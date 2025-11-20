import React from 'react';
import { Box, Typography, useTheme } from '@mui/material';
import { APP_THEMES, FOOTER_DISCLAIMER_TEXT } from '../constants';

const Footer = ({ appTheme = APP_THEMES.DARK }) => {
    const theme = useTheme();

    const footerStyles = {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        zIndex: 1000,
        backgroundColor: 'transparent',
        padding: theme.spacing(1, 2),
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: theme.spacing(1),
    };

    const textStyles = {
        fontSize: '0.75rem',
        color: theme.palette.text.secondary,
        backgroundColor: appTheme === APP_THEMES.DARK 
        ? 'rgba(0, 0, 0, 0.35)' 
        : 'rgba(255, 255, 255, 0.5)',
        padding: theme.spacing(0.5, 1.5),
        borderRadius: theme.shape.borderRadius || 4,
        backdropFilter: 'blur(4px)',
    };

    return (
        <Box sx={footerStyles}>
        <Typography sx={textStyles}>
            {FOOTER_DISCLAIMER_TEXT}
        </Typography>
        </Box>
    );
};

export default Footer;

