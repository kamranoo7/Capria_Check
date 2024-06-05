from functools import wraps
from flask import render_template, session,redirect,url_for
import threading
from admin import (
    updateAdminFlag,
    getAdminFlag,
    updateAdminLastActivity,
    getAdminLastActivity,
)
from datetime import datetime, timezone








running_thread = None

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'userID' not in session:
            return render_template('login.html', message='You need to login first!!')
        
        if getAdminFlag()==True and session['role']=='user':
            return render_template('login.html', message="Sorry for inconvenience. Website went under maintenance. Please contact Admin.")
        
        if getAdminFlag()==False and session['role']=='admin':
            updateAdminFlag(True)
            time = datetime.now(timezone.utc)
            print(time,'time ---------->>>>>>')
            updateAdminLastActivity(time) 
        
        return f(*args, **kwargs)
    return decorated_function



def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'userID' in session and session['role'] == 'admin':
            if getAdminFlag()==False:
                updateAdminFlag(True)
                updateAdminLastActivity(datetime.now(timezone.utc))
            return f(*args, **kwargs)
        else:
            return render_template('login.html',message='Only Admin can access this route!!!')
    return decorated_function




def user_activity_tracker(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session['role']=='admin':
            updateAdminLastActivity(datetime.now(timezone.utc)) 
        return f(*args, **kwargs)
    return decorated_function








# def user_activity_tracker(f):
#     @wraps(f)
#     def decorated_function(*args, **kwargs):
#         if session['role']=='admin':
#             last_activity_time = session.get('last_activity_time')
#             check_for_admin_activity() 
#             if last_activity_time:
#                 last_activity_time = last_activity_time.replace(tzinfo=timezone.utc)
#                 time_diff = datetime.now(timezone.utc) - last_activity_time
#                 if time_diff.total_seconds() > INACTIVE_TIMEOUT:
#                     session.clear()
#                     resetAdminCounter()
#                     return render_template('login.html', message='You have been logged out due to inactivity.')
#         session['last_activity_time'] = datetime.now(timezone.utc)
#         return f(*args, **kwargs)
#     return decorated_function



    
