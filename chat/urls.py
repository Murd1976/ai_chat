from django.urls import include, path, re_path
from chat import views
from .views import BBLoginView, BBLogoutView

app_name = 'chat'
urlpatterns = [
    path('', views.index, name='index'),
    path('chat/', views.chat_home),
    #path('chat/gpt/', views.gpt_response, name= 'my_chat_gpt'),
    path('chat/gpt/', views.chat_page, name= 'my_chat_gpt'),
    path('chat/db/', views.load_base_page, name= 'my_load_base'),
    path('chat/db/train/', views.train_page, name= 'my_train_model'),
    path('chat/db/train/status/', views.train_status_page, name= 'my_status_model'),
    path('chat/db/new/<int:id>/', views.new_proposal_page, name= 'my_new_proposal'),
    path('chat/db/new/', views.new_proposal_page, name= 'my_new_proposal'),
    path('chat/db/edit/<int:id>/', views.edit_proposal_page, name= 'my_edit_proposal'),
    path('chat/db/edit/', views.edit_proposal_page, name= 'my_edit_proposal'),
    
    path('chat/db/edit/del/<int:id>/', views.delete_record_edit, name= 'my_del_record_edit'),
    path('chat/db/new/del/<int:id>/', views.delete_record_new, name= 'my_del_record_new'),
    
    
    
    path('accounts/register/activate/<str:sign>/', views.user_activate, name= 'my_register_activate'),
    path('accounts/register/done/', views.RegisterDoneView.as_view(), name= 'my_register_done'),
    path('accounts/register/', views.RegisterUserView.as_view(), name= 'my_register'),
    path('accounts/profile/delete/', views.DeleteUserView.as_view(), name= 'my_profile_delete'),
    path('accounts/profile/change/', views.ChangeUserInfoView.as_view(), name= 'my_profile_change'),
    path('accounts/profile/', views.user_profile, name= 'my_profile'),
    path('accounts/login/', BBLoginView.as_view(), name= 'my_login'),
    path('accounts/logout/', BBLogoutView.as_view(), name= 'my_logout'),
    path('accounts/password/change/', views.BBPasswordChangeView.as_view(), name= 'my_password_change'),
    path('<str:page>/', views.other_page, name= 'other'),
    
    
    path('m400', views.m400),
    
     
]



