import { CONFIG } from '../config';

/**
 * Logger utility that respects the development environment
 * Only logs when IS_DEV_STAGE is true
 */
export const logger = {
  log: (...args) => {
    if (CONFIG.IS_DEV_STAGE) {
      console.log(...args);
    }
  },

  warn: (...args) => {
    if (CONFIG.IS_DEV_STAGE) {
      console.warn(...args);
    }
  },

  error: (...args) => {
    if (CONFIG.IS_DEV_STAGE) {
      console.error(...args);
    }
  },

  info: (...args) => {
    if (CONFIG.IS_DEV_STAGE) {
      console.info(...args);
    }
  },

  debug: (...args) => {
    if (CONFIG.IS_DEV_STAGE) {
      console.debug(...args);
    }
  }
};
