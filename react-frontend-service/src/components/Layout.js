import React from 'react';
import { Outlet } from 'react-router-dom';
import { Header } from '../components/Header';

const Layout = () => {
    return (
        <>
            <Header />
            <div className="container mt-4">
                <Outlet />
            </div>
        </>
    );
};

export default Layout;
