import React from 'react';
import {
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    DialogContentText,
    Button,
    Box,
    Paper,
    Typography
} from '@mui/material';
import { HELP_DIALOG } from '../constants';

const SupportDialog = ({ open, onClose }) => {
    return (
        <Dialog
        open={open}
        onClose={onClose}
        maxWidth="md"
        fullWidth
        >
        <DialogTitle>{HELP_DIALOG.TITLE}</DialogTitle>
        <DialogContent>
            <DialogContentText sx={{ mb: 3 }}>
            {HELP_DIALOG.DESCRIPTION}
            </DialogContentText>

            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Paper elevation={0} sx={{ p: 2, border: '1px solid', borderColor: 'divider' }}>
                <Typography variant="h6" gutterBottom>
                {HELP_DIALOG.SECTIONS.DOCUMENTATION.TITLE}
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                {HELP_DIALOG.SECTIONS.DOCUMENTATION.DESCRIPTION}
                </Typography>
                <Button
                variant="outlined"
                size="small"
                href={HELP_DIALOG.SECTIONS.DOCUMENTATION.URL}
                target="_blank"
                rel="noopener noreferrer"
                >
                {HELP_DIALOG.SECTIONS.DOCUMENTATION.BUTTON}
                </Button>
            </Paper>

            <Paper elevation={0} sx={{ p: 2, border: '1px solid', borderColor: 'divider' }}>
                <Typography variant="h6" gutterBottom>
                {HELP_DIALOG.SECTIONS.CONTACT_SUPPORT.TITLE}
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                {HELP_DIALOG.SECTIONS.CONTACT_SUPPORT.DESCRIPTION}
                </Typography>
                <Typography variant="body2" sx={{ mb: 2, fontWeight: 'medium' }}>
                Email: <a href={`mailto:${HELP_DIALOG.SECTIONS.CONTACT_SUPPORT.EMAIL}`} style={{ color: 'inherit' }}>{HELP_DIALOG.SECTIONS.CONTACT_SUPPORT.EMAIL}</a>
                </Typography>
                <Button
                variant="outlined"
                size="small"
                href={`mailto:${HELP_DIALOG.SECTIONS.CONTACT_SUPPORT.EMAIL}?subject=InstaGeo Support Request`}
                >
                {HELP_DIALOG.SECTIONS.CONTACT_SUPPORT.BUTTON}
                </Button>
            </Paper>
            </Box>
        </DialogContent>
        <DialogActions>
            <Button onClick={onClose} variant="contained">
            {HELP_DIALOG.BUTTONS.CLOSE}
            </Button>
        </DialogActions>
        </Dialog>
    );
    };

    export default SupportDialog;

