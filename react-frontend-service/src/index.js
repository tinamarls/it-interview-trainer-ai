import React from 'react';
import ReactDOM from 'react-dom/client';
import { RouterProvider } from 'react-router-dom';
import { router } from './api/router';

import 'bootstrap/dist/css/bootstrap.min.css';
import { AuthProvider } from './pages/AuthProvider';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
    <AuthProvider>
        <div style={{display: 'flex', justifyContent: 'center'}}>
            <RouterProvider router={router} />
        </div>
    </AuthProvider>
);

