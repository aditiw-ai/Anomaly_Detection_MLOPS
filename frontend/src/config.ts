export const config = {
    API_BASE_URL: import.meta.env.VITE_API_BASE_URL as string,
    ENV: import.meta.env.MODE as string,
};

// Named export for convenience
export const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL;