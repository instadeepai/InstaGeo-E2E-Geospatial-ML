/**
 * Utility functions for authentication-related errors.
 */

export const isAuthenticationError = (error) => {
  if (!error) return false;

  const errorMessage = error?.message || error;
  const message = typeof errorMessage === 'string' ? errorMessage : String(errorMessage);
  const lowerMessage = message.toLowerCase();

  return (
    lowerMessage.includes('not authenticated') ||
    lowerMessage.includes('login required') ||
    lowerMessage.includes('sign in') ||
    lowerMessage.includes('authentication') ||
    lowerMessage.includes('401') ||
    lowerMessage.includes('403') ||
    message.includes('Token invalid') ||
    message.includes('Token validation failed')
  );
};
